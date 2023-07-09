from typing import Sequence

from xdsl.dialects import builtin, riscv, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import MLContext, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def cast_values_to_registers(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[OpResult]:
    if not operands:
        return []
    types = [riscv.RegisterType(riscv.Register()) for _ in range(len(operands))]
    cast = builtin.UnrealizedConversionCastOp.get(operands, types)
    rewriter.insert_op_before_matched_op(cast)
    return cast.results


class LowerForOp(RewritePattern):
    counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter):
        c = self.counter
        self.counter += 1

        lb, ub, step = cast_values_to_registers(op.operands, rewriter)

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
