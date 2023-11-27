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


def get_snitch_reserved() -> set[riscv.FloatRegisterType]:
    """
    Utility method to make explicit the Snitch ISA assumptions wrt the
    floating-point registers that are considered reserved.
    Currently, the first 3 floating-point registers are reserved.
    """

    num_reserved = 3
    assert len(riscv.Registers.FT) >= num_reserved

    return {riscv.Registers.FT[i] for i in range(0, num_reserved)}


class AllocateSnitchStridedStreamRegisters(RewritePattern):
    """
    Allocates the register used by the stream as the one specified by the `dm`
    (data mover) attribute. Must be called before allocating the registers in the
    `snitch_stream.generic` body.
    """

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
    """
    Allocates the registers in the body of a `snitch_stream.generic` operation by assigning
    them to the ones specified by the streams.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.GenericOp, rewriter: PatternRewriter, /
    ):
        block = op.body.block

        for arg, input in zip(block.args, op.inputs):
            assert isinstance(input_type := input.type, stream.ReadableStreamType)
            rs_input_type: stream.ReadableStreamType[Any] = input_type
            arg.type = rs_input_type.element_type

        yield_op = block.last_op
        assert isinstance(yield_op, snitch_stream.YieldOp)

        for arg, output in zip(yield_op.values, op.outputs):
            assert isinstance(output_type := output.type, stream.WritableStreamType)
            rs_output_type: stream.WritableStreamType[Any] = output_type
            arg.type = rs_output_type.element_type


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
