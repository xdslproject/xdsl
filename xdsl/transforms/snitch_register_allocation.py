from dataclasses import dataclass
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import riscv, riscv_snitch, snitch_stream, stream
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
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


class AllocateSnitchStreamingRegionRegisters(RewritePattern):
    """
    Allocates the registers in the body of a `snitch_stream.streaming_region` operation by
    assigning them to the ones specified by the streams.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StreamingRegionOp, rewriter: PatternRewriter, /
    ):
        block = op.body.block

        for index, input_stream in enumerate(block.args):
            input_stream.type = stream.ReadableStreamType(riscv.Registers.FT[index])

        input_count = len(op.inputs)

        for index, output_stream in enumerate(block.args[input_count:]):
            output_stream.type = stream.WritableStreamType(
                riscv.Registers.FT[index + input_count]
            )


class AllocateRiscvSnitchReadRegisters(RewritePattern):
    """
    Propagates the register allocation done at the stream level to the values read from
    the streams.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_snitch.ReadOp, rewriter: PatternRewriter, /):
        stream_type = cast(
            stream.ReadableStreamType[riscv.FloatRegisterType], op.stream.type
        )
        op.res.type = stream_type.element_type


class AllocateRiscvSnitchWriteRegisters(RewritePattern):
    """
    Propagates the register allocation done at the stream level to the values written to
    the streams.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_snitch.WriteOp, rewriter: PatternRewriter, /):
        stream_type = cast(
            stream.WritableStreamType[riscv.FloatRegisterType], op.stream.type
        )
        op.value.type = stream_type.element_type


@dataclass(frozen=True)
class SnitchRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers for snitch operations.
    """

    name = "snitch-allocate-registers"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            AllocateSnitchStreamingRegionRegisters(),
            apply_recursively=False,
        ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AllocateRiscvSnitchReadRegisters(),
                    AllocateRiscvSnitchWriteRegisters(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
