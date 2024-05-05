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
        if inner_lb.value.data != 0:
            return

        if (inner_ub := get_constant_value(inner_loop.ub)) is None:
            return
        if (outer_step := get_constant_value(op.step)) is None:
            return
        if inner_ub != outer_step:
            return

        outer_index = outer_body.args[0]
        inner_index = inner_loop.body.block.args[0]

        if len(outer_index.uses) != 1 or len(inner_index.uses) != 1:
            # If the induction variable is used more than once, we can't fold it
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
        moved_region = rewriter.move_region_contents_to_new_regions(inner_loop.body)
        rewriter.erase_op(inner_loop)

        rewriter.replace_matched_op(
            riscv_scf.ForOp(
                op.lb,
                op.ub,
                inner_loop.step,
                (),
                moved_region,
            )
        )


class RiscvScfLoopFusionPass(ModulePass):
    """
    Folds loop nests if they can be represented with a single loop.
    Currently does this by matching the inner loop range with the outer loop step.
    If the inner iteration space fits perfectly in the outer iteration step, then merge.
    """

    name = "riscv-scf-loop-fusion"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FuseNestedLoopsPattern()).rewrite_module(op)
