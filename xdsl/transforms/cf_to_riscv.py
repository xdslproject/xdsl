from abc import ABC
from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import cf
from xdsl.dialects import riscv
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)


class LowerConditionalBranchToRISCV(RewritePattern, ABC):
    """ """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.ConditionalBranch, rewriter: PatternRewriter, /):
        if parent_region := op.then_block.parent:
            then_label = riscv.LabelOp(
                str(parent_region.get_block_index(op.then_block))
            )
            else_label = riscv.LabelOp(
                str(parent_region.get_block_index(op.else_block))
            )

            zero = riscv.GetRegisterOp(riscv.Registers.ZERO).res
            cond = riscv.GetRegisterOp(riscv.Registers.T0).res
            op.cond.replace_by(cond)
            branch = riscv.BeqOp(cond, zero, then_label.label)

            rewriter.insert_op_at_start(then_label, op.then_block)
            rewriter.insert_op_at_start(else_label, op.else_block)

            rewriter.replace_matched_op(branch)


class CfToRISCV(ModulePass):
    """ """

    name = "cf-to-riscv"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerConditionalBranchToRISCV()])
        ).rewrite_module(op)
