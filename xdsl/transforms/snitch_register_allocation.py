from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import riscv, riscv_snitch, snitch, snitch_stream
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


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
            rewriter.replace_value_with_new_type(
                input_stream, snitch.ReadableStreamType(riscv.Registers.FT[index])
            )

        input_count = len(op.inputs)

        for index, output_stream in enumerate(block.args[input_count:]):
            rewriter.replace_value_with_new_type(
                output_stream,
                snitch.WritableStreamType(riscv.Registers.FT[index + input_count]),
            )


class AllocateRiscvSnitchReadRegisters(RewritePattern):
    """
    Propagates the register allocation done at the stream level to the values read from
    the streams.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_snitch.ReadOp, rewriter: PatternRewriter, /):
        stream_type = cast(
            snitch.ReadableStreamType[riscv.FloatRegisterType], op.stream.type
        )
        rewriter.replace_value_with_new_type(op.res, stream_type.element_type)


class AllocateRiscvSnitchWriteRegisters(RewritePattern):
    """
    Propagates the register allocation done at the stream level to the values written to
    the streams.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_snitch.WriteOp, rewriter: PatternRewriter, /):
        stream_type = cast(
            snitch.WritableStreamType[riscv.FloatRegisterType], op.stream.type
        )
        rewriter.replace_value_with_new_type(op.value, stream_type.element_type)


@dataclass(frozen=True)
class SnitchRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers for snitch operations.
    """

    name = "snitch-allocate-registers"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
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
