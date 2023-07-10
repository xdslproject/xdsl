from abc import ABC

from xdsl.dialects import cf, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
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

            cond = UnrealizedConversionCastOp.get(
                [op.cond], [riscv.RegisterType(riscv.Register())]
            )
            print(cond)
            zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
            branch = riscv.BeqOp(cond.results[0], zero, then_label.label)
            print(branch)

            print(op.then_block.parent)
            # then_block = op.region.detach_block(op.then_block)
            # else_block = op.region.detach_block(op.else_block)

            # rewriter.insert_op_at_start(then_label, then_block)
            # rewriter.insert_op_at_start(else_label, else_block)

            rewriter.replace_matched_op([cond, zero, branch])


class CfToRISCV(ModulePass):
    """ """

    name = "cf-to-riscv"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerConditionalBranchToRISCV()])
        ).rewrite_module(op)
