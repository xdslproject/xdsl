"""
Rewrite patterns for lowering snitch → riscv.
"""

from typing import Tuple
from dataclasses import dataclass
from xdsl.passes import ModulePass

from xdsl.ir import MLContext
from xdsl.irdl import Operand

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)

from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.dialects import snitch, riscv, builtin


@dataclass(frozen=True)
class SnitchStreamerDimension:
    # Offset of the streamer bound configuration for a specific dimension
    Bound: int

    # Offset of the streamer stride configuration for a specific dimension
    Stride: int

    # Offset of the streamer source address configuration for a specific dimension
    Source: int

    # Offset of the streamer source address configuration for a specific dimension
    Destination: int


@dataclass(frozen=True)
class SnitchStreamerMemoryMap:
    """
    In the Snitch architecture, each streamer (a.k.a. data mover)
    is configured via a memory-mapped address space that can be written
    via custom riscv.scfgw (Stream ConFiGure Write) operation. For each
    streamer we have:

    * Repeat: how many times a value should be repeated when
              popped from/pushed to a stream
    * Dimensions: a list of supported streaming dimensions

    For each dimension, the supported configuration parameters are:

    * Bound
    * Stride
    * Source: base address when reading from a stream
    * Destination: base address when writing to a stream

    This table encodes the base addresses for each of the configuration
    parameters above.
    """

    # Global streaming behaviour enable/disable register.
    # Accessible as a regular RISC-V CSR.
    Csr: int = 0x7C0

    # Offset of the streamer repetition configuration.
    Repeat: int = 0x01

    Dimension: Tuple[SnitchStreamerDimension, ...] = (
        # Dimension 0
        SnitchStreamerDimension(
            0x02,  # Bound
            0x06,  # Stride
            0x18,  # Source
            0x1C,  # Destination
        ),
        # Dimension 1
        SnitchStreamerDimension(
            0x03,  # Bound
            0x07,  # Stride
            0x19,  # Source
            0x1D,  # Destination
        ),
        # Dimension 2
        SnitchStreamerDimension(
            0x04,  # Bound
            0x08,  # Stride
            0x1A,  # Source
            0x1E,  # Destination
        ),
        # Dimension 3
        SnitchStreamerDimension(
            0x05,  # Bound
            0x09,  # Stride
            0x1B,  # Source
            0x1F,  # Destination
        ),
    )


def make_stream_set_config_ops(value: Operand, stream: Operand, baseaddr: int):
    """
    Return the list of riscv operations needed to set a specific SSR configuration
    parameter located at 'baseaddr' to a specific 'value' for a specific data mover
    identified by 'stream'.

    To compute the actual address of the memory-mapped configuration parameter,
    we have to compute:

    address = stream + baseaddr << 5

    This value is then passed to riscv.scfgw to perform the actual setting.
    """
    return [
        address := riscv.AddiOp(
            stream,
            immediate=IntegerAttr(baseaddr << 5, i32),
        ),
        riscv.ScfgwOp(
            rs1=value,
            rs2=address,
        ),
    ]


class LowerSsrSetDimensionBoundOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionBoundOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.value.data
        ops = make_stream_set_config_ops(
            value=op.value,
            stream=op.stream,
            baseaddr=SnitchStreamerMemoryMap.Dimension[dim].Bound,
        )
        rewriter.replace_matched_op(
            [*ops],
            [],
        )


class LowerSsrSetDimensionStrideOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionStrideOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.value.data
        ops = make_stream_set_config_ops(
            value=op.value,
            stream=op.stream,
            baseaddr=SnitchStreamerMemoryMap.Dimension[dim].Stride,
        )
        rewriter.replace_matched_op(
            [*ops],
            [],
        )


class LowerSsrSetDimensionSourceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionSourceOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.value.data
        ops = make_stream_set_config_ops(
            value=op.value,
            stream=op.stream,
            baseaddr=SnitchStreamerMemoryMap.Dimension[dim].Source,
        )
        rewriter.replace_matched_op(
            [*ops],
            [],
        )


class LowerSsrSetDimensionDestinationOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionDestinationOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.value.data
        ops = make_stream_set_config_ops(
            value=op.value,
            stream=op.stream,
            baseaddr=SnitchStreamerMemoryMap.Dimension[dim].Destination,
        )
        rewriter.replace_matched_op(
            [*ops],
            [],
        )


class LowerSsrSetStreamRepetitionOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetStreamRepetitionOp, rewriter: PatternRewriter, /
    ):
        ops = make_stream_set_config_ops(
            value=op.value,
            stream=op.stream,
            baseaddr=SnitchStreamerMemoryMap.Repeat,
        )
        rewriter.replace_matched_op(
            [*ops],
            [],
        )


class LowerSsrEnable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrEnable, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            [
                riscv.CsrrsiOp(
                    csr=IntegerAttr(SnitchStreamerMemoryMap.Csr, i32),
                    immediate=IntegerAttr(1, i32),
                    rd=riscv.Registers.ZERO,
                )
            ],
            [],
        )


class LowerSsrDisable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrDisable, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            [
                riscv.CsrrciOp(
                    csr=IntegerAttr(SnitchStreamerMemoryMap.Csr, i32),
                    immediate=IntegerAttr(1, i32),
                    rd=riscv.Registers.ZERO,
                )
            ],
            [],
        )


@dataclass
class LowerSnitchPass(ModulePass):
    name = "lower-snitch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerSsrSetDimensionBoundOp(),
                    LowerSsrSetDimensionStrideOp(),
                    LowerSsrSetDimensionSourceOp(),
                    LowerSsrSetDimensionDestinationOp(),
                    LowerSsrSetStreamRepetitionOp(),
                    LowerSsrEnable(),
                    LowerSsrDisable(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
