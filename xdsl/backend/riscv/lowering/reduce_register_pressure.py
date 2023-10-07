from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LoadZeroImmediatePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.LiOp, rewriter: PatternRewriter) -> None:
        if not isinstance(op.immediate, IntegerAttr) or op.immediate.value.data != 0:
            return

        assert isinstance(op.rd.type, riscv.IntRegisterType)

        if op.rd.type.is_allocated:
            rewriter.replace_matched_op(
                (
                    zero_op := riscv.GetRegisterOp(riscv.Registers.ZERO),
                    riscv.MVOp(zero_op.res, rd=op.rd.type),
                )
            )
        else:
            rewriter.replace_matched_op(riscv.GetRegisterOp(riscv.Registers.ZERO))


class RiscvReduceRegisterPressurePass(ModulePass):
    name = "riscv-reduce-register-pressure"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LoadZeroImmediatePattern(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
