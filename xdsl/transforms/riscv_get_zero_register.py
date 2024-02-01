from dataclasses import dataclass
from typing import cast

from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class GetZeroRegisterPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.LiOp, rewriter: PatternRewriter) -> None:
        if not (isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0):
            return

        reg = cast(riscv.IntRegisterType, op.rd.type)

        if reg == riscv.Registers.ZERO:
            rewriter.replace_matched_op(riscv.GetRegisterOp(reg))
        else:
            rewriter.replace_matched_op(
                (
                    zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                    riscv.MVOp(zero.res, rd=reg),
                )
            )


@dataclass
class RiscvGetZeroRegisterPass(ModulePass):
    """
    Converts li 0 to more efficient loading from zero register.
    """

    name = "riscv-get-zero-register"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([GetZeroRegisterPattern()]),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
