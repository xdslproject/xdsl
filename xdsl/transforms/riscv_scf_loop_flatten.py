from typing import cast

from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value

#  This pass flattens pairs nested loops into a single loop.
#
#  Similar to LLVM's loop flattening:
#  https://llvm.org/doxygen/LoopFlatten_8cpp_source.html
#
#  The intention is to optimise loop nests like this, which together access an
#  array linearly:
#
#    for i in range(N):
#      for j in range(M):
#        f(A[i*M+j])
#
#    for o in range(ol, ou, os):
#      for i in range(il, iu, is):
#        # neither o nor i are used
#
#    These become:
#    for i in range(0, N * M, k):
#      for j in range(k):
#        f(A[i+j])
#
#    factor = (iu - il) // is
#    for o in range(ol, ou * factor, os):
#      # o is not used
#


class FuseNestedLoopsPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter) -> None:
        if op.iter_args:
            return

        outer_body = op.body.block
        if not isinstance(inner_loop := outer_body.first_op, riscv_scf.ForOp):
            # Outer loop must contain inner loop
            return
        if inner_loop is not cast(riscv_scf.YieldOp, outer_body.last_op).prev_op:
            # Outer loop must contain only inner loop and yield
            return
        if inner_loop.iter_args:
            return

        if (inner_lb := get_constant_value(inner_loop.lb)) is None:
            return

        if (inner_ub := get_constant_value(inner_loop.ub)) is None:
            return
        if (outer_step := get_constant_value(op.step)) is None:
            return

        outer_index = outer_body.args[0]
        inner_index = inner_loop.body.block.args[0]

        if outer_index.uses or inner_index.uses:
            if inner_lb.value.data != 0:
                return

            if inner_ub != outer_step:
                return

            # If either induction variable is used, we can only fold if used exactly once
            if len(outer_index.uses) != 1 or len(inner_index.uses) != 1:
                return

            outer_user = next(iter(outer_index.uses)).operation
            inner_user = next(iter(inner_index.uses)).operation
            if outer_user is not inner_user:
                return

            user = outer_user

            if not isinstance(user, riscv.AddOp):
                return

            # We can fuse
            user.rd.replace_by(inner_index)
            rewriter.erase_op(user)
            new_ub = op.ub
            new_step = inner_loop.step
        else:
            if (inner_step := get_constant_value(inner_loop.step)) is None:
                return

            if (outer_lb := get_constant_value(op.lb)) is None:
                return

            if outer_lb.value.data != 0:
                # Do not currently handle lb != 0
                return

            factor = (
                inner_ub.value.data - inner_lb.value.data
            ) // inner_step.value.data
            factor_op = riscv.LiOp(factor, rd=riscv.Registers.UNALLOCATED_INT)
            new_ub_op = riscv.MulOp(
                op.ub, factor_op.rd, rd=riscv.Registers.UNALLOCATED_INT
            )
            rewriter.insert_op_before_matched_op((factor_op, new_ub_op))
            new_ub = new_ub_op.rd
            new_step = op.step

        moved_region = rewriter.move_region_contents_to_new_regions(inner_loop.body)
        rewriter.erase_op(inner_loop)

        rewriter.replace_matched_op(
            riscv_scf.ForOp(
                op.lb,
                new_ub,
                new_step,
                (),
                moved_region,
            )
        )


class RiscvScfLoopFlattenPass(ModulePass):
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

    name = "riscv-scf-loop-flatten"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FuseNestedLoopsPattern()).rewrite_module(op)
