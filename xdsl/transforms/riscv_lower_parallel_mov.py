from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ParallelMovPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.ParallelMovOp, rewriter: PatternRewriter):
        input_types: Sequence[riscv.RISCVRegisterType] = op.inputs.types
        output_types: Sequence[riscv.RISCVRegisterType] = op.outputs.types

        assert all(i.is_allocated for i in input_types), (
            "All registers must be allocated"
        )
        assert all(i.is_allocated for i in output_types), (
            "All registers must be allocated"
        )

        results = []  # The resulting SSA values to use
        for src, dst in zip(op.inputs, op.outputs, strict=True):
            if src.type == dst.type:
                results.append(src)
            else:
                raise NotImplementedError

        rewriter.replace_matched_op(tuple(), results)


@dataclass(frozen=True)
class RISCVLowerParallelMovPass(ModulePass):
    """Lowers ParallelMovOp in a module into separate moves."""

    name = "riscv-lower-parallel-mov"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ParallelMovPattern()).rewrite_module(op)
