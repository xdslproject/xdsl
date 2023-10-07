from dataclasses import dataclass

from xdsl.dialects import riscv, snitch_stream, stream
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AllocateSnitchGenericRegisters(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.GenericOp, rewriter: PatternRewriter, /
    ):
        block = op.body.block
        yield_op = block.last_op
        assert isinstance(yield_op, snitch_stream.YieldOp)

        for a, arg in enumerate(block.args):
            arg.type = riscv.Registers.FT[a]

        for i, arg in enumerate(op.inputs):
            arg.type = stream.InputStreamType(riscv.Registers.FT[i])

        input_count = len(op.inputs)

        for o, arg in enumerate(op.outputs):
            t = riscv.Registers.FT[o + input_count]
            arg.type = stream.OutputStreamType(t)
            yield_op.operands[o].type = t


@dataclass
class SnitchRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers for snitch operations.
    """

    name = "snitch-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            AllocateSnitchGenericRegisters(),
            apply_recursively=False,
        ).rewrite_module(op)
