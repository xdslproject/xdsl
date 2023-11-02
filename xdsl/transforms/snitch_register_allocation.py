from dataclasses import dataclass
from typing import Any, cast

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


class AllocateSnitchStridedStreamRegisters(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: snitch_stream.StridedReadOp | snitch_stream.StridedWriteOp,
        rewriter: PatternRewriter,
        /,
    ):
        stream_type = op.stream.type
        assert isinstance(
            stream_type, stream.ReadableStreamType | stream.WritableStreamType
        )
        stream_type = cast(stream.StreamType[Any], stream_type)
        op.stream.type = type(stream_type)(riscv.Registers.FT[op.dm.data])


class AllocateSnitchGenericRegisters(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.GenericOp, rewriter: PatternRewriter, /
    ):
        block = op.body.block

        for arg, input in zip(block.args, op.inputs):
            assert isinstance(input.type, stream.ReadableStreamType)
            input_type: stream.ReadableStreamType[Any] = input.type
            arg.type = input_type.element_type

        yield_op = block.last_op
        assert isinstance(yield_op, snitch_stream.YieldOp)

        for arg, output in zip(yield_op.values, op.outputs):
            assert isinstance(output.type, stream.WritableStreamType)
            output_type: stream.WritableStreamType[Any] = output.type
            arg.type = output_type.element_type


@dataclass
class SnitchRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers for snitch operations.
    """

    name = "snitch-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            AllocateSnitchStridedStreamRegisters(),
            apply_recursively=False,
        ).rewrite_module(op)
        PatternRewriteWalker(
            AllocateSnitchGenericRegisters(),
            apply_recursively=False,
        ).rewrite_module(op)
