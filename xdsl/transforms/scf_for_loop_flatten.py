from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand

#  This pass flattens pairs nested loops into a single loop.
#
#  Similar to LLVM's loop flattening:
#  https://llvm.org/doxygen/LoopFlatten_8cpp_source.html
#
#  The intention is to optimise loop nests like this, which together access an
#  array linearly:
#
#    for i in range(x, N, M):
#      for j in range(0, M, K):
#        f(A[i+j])
#
#    for o in range(ol, ou, os):
#      for i in range(il, iu, is):
#        # neither o nor i are used
#
#    These become:
#    # (If K is constant and divides M)
#    for i in range(x, N, K):
#      f(A[i])
#
#    factor = (iu - il) // is
#    for o in range(ol, ou * factor, os):
#      # o is not used
#


class FlattenNestedLoopsPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        outer_body = op.body.block
        if not isinstance(inner_loop := outer_body.first_op, scf.ForOp):
            # Outer loop must contain inner loop
            return
        if (
            inner_loop
            is not (outer_yield_op := cast(scf.YieldOp, outer_body.last_op)).prev_op
        ):
            # Outer loop must contain only inner loop and yield
            return

        if op.iter_args:
            if not inner_loop.iter_args:
                return
            if len(op.iter_args) != len(inner_loop.iter_args):
                return
            if not all(
                lhs is rhs
                for (lhs, rhs) in zip(outer_body.args[1:], inner_loop.iter_args)
            ):
                return
            if not all(
                lhs is rhs
                for (lhs, rhs) in zip(inner_loop.results, outer_yield_op.operands)
            ):
                return
        elif inner_loop.iter_args:
            return

        if (inner_lb := const_evaluate_operand(inner_loop.lb)) is None:
            return

        if (inner_ub := const_evaluate_operand(inner_loop.ub)) is None:
            return
        if (outer_step := const_evaluate_operand(op.step)) is None:
            return
        if (inner_step := const_evaluate_operand(inner_loop.step)) is None:
            return

        outer_index = outer_body.args[0]
        inner_index = inner_loop.body.block.args[0]

        if outer_index.uses or inner_index.uses:
            if inner_lb != 0:
                return

            if inner_ub != outer_step:
                return

            if outer_step % inner_step:
                return

            # If either induction variable is used, we can only fold if used exactly once
            if not outer_index.has_one_use() or not inner_index.has_one_use():
                return

            outer_user = next(iter(outer_index.uses)).operation
            inner_user = next(iter(inner_index.uses)).operation
            if outer_user is not inner_user:
                return

            user = outer_user

            if not isinstance(user, arith.AddiOp):
                return

            # We can fuse
            user.result.replace_by(inner_index)
            rewriter.erase_op(user)
            new_ub = op.ub
            new_step = inner_loop.step
        else:
            if (outer_lb := const_evaluate_operand(op.lb)) is None:
                return

            if outer_lb != 0:
                # Do not currently handle lb != 0
                return

            factor = (inner_ub - inner_lb) // inner_step
            factor_op = arith.ConstantOp(
                builtin.IntegerAttr(factor, builtin.IndexType())
            )
            new_ub_op = arith.MuliOp(op.ub, factor_op.result)
            rewriter.insert_op((factor_op, new_ub_op))
            new_ub = new_ub_op.result
            new_step = op.step

        moved_region = rewriter.move_region_contents_to_new_regions(inner_loop.body)
        rewriter.erase_op(outer_yield_op)
        rewriter.erase_op(inner_loop)

        rewriter.replace_op(
            op,
            scf.ForOp(
                op.lb,
                new_ub,
                new_step,
                op.iter_args,
                moved_region,
            ),
        )


class ScfForLoopFlattenPass(ModulePass):
    """
    Folds perfect loop nests if they can be represented with a single loop.
    Currently does this by matching the inner loop range with the outer loop step.
    If the inner iteration space fits perfectly in the outer iteration step, then merge.
    Other conditions:
     - the only use of the induction arguments must be an add operation, this op is fused
       into a single induction argument,
     - the lower bound of the inner loop must be 0,
     - the loops must have no iteration arguments.
    """

    name = "scf-for-loop-flatten"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FlattenNestedLoopsPattern()).rewrite_module(op)
