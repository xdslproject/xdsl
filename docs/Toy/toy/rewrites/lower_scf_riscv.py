from xdsl.dialects import riscv, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from .lower_riscv_cf import cast_value_to_register


class LowerForOp(RewritePattern):
    counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter):
        c = self.counter
        self.counter += 1

        lb = cast_value_to_register(op.lb, rewriter)
        ub = cast_value_to_register(op.ub, rewriter)
        step = cast_value_to_register(op.step, rewriter)

        rewriter.insert_op_before_matched_op(
            [
                index := riscv.MVOp(lb, comment="i = lb"),
                body_label := riscv.LabelOp(f"for_body_{c}"),
                riscv.BeqOp(index, ub, riscv.LabelAttr(f"for_end_{c}")),
            ]
        )

        assert len(op.body.block.args) == 1

        op.body.block.args[0].replace_by(index.rd)
        rewriter.inline_block_before_matched_op(op.body.block)

        rewriter.replace_matched_op(
            [
                riscv.AddOp(index, step, rd_operand=index, comment="i += step"),
                riscv.JOp(body_label.label),
                riscv.LabelOp(f"for_end_{c}"),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.Yield, rewriter: PatternRewriter):
        assert not op.operands
        rewriter.erase_matched_op()


class LowerScfRiscvPass(ModulePass):
    name = "lower-scf-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerForOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
