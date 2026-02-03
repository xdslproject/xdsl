"""
Rewrite patterns for lowering snitch → riscv.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_snitch, snitch
from xdsl.dialects.builtin import IntegerAttr, i32
from xdsl.ir import Operation
from xdsl.irdl import Operand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass(frozen=True)
class SnitchStreamerDimension:
    # Offset of the streamer bound configuration for a specific dimension
    bound: int

    # Offset of the streamer stride configuration for a specific dimension
    stride: int

    # Offset of the streamer source address configuration for a specific dimension
    source: int

    # Offset of the streamer source address configuration for a specific dimension
    destination: int


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
    csr: int = 0x7C0

    # Offset of the streamer repetition configuration.
    repeat: int = 0x01

    dimension: tuple[SnitchStreamerDimension, ...] = (
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


def write_ssr_config_ops(
    reg: int, dm: int, value: Operand, comment: str | None = None
) -> Iterable[Operation]:
    """
    Return the list of riscv operations needed to set a specific SSR configuration
    parameter located at 'reg' to a specific 'value' for a specific data mover
    identified by 'dm'.

    To compute the actual address of the memory-mapped configuration parameter,
    we have to compute:

    address = dm + reg << 5

    This value is then passed to riscv.scfgw to perform the actual setting.

    Reference implementation in the snitch runtime library:
    ```C
    inline void write_ssr_cfg(uint32_t reg, uint32_t dm, uint32_t value) {
        asm volatile("scfgwi %[value], %[dm] | %[reg]<<5\n" ::[value] "r"(value),
                    [ dm ] "i"(dm), [ reg ] "i"(reg));
    }
    ```
    """
    return [
        address := riscv.LiOp(immediate=IntegerAttr(dm | reg << 5, i32)),
        riscv_snitch.ScfgwOp(rs1=value, rs2=address, comment=comment),
    ]


class LowerSsrSetDimensionBoundOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionBoundOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.data
        ops = write_ssr_config_ops(
            dm=op.dm.data,
            reg=SnitchStreamerMemoryMap.dimension[dim].bound,
            value=op.value,
            comment=f"dm {op.dm.data} dim {dim} bound",
        )
        rewriter.replace_op(
            op,
            [*ops],
            [],
        )


class LowerSsrSetDimensionStrideOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionStrideOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.data
        ops = write_ssr_config_ops(
            dm=op.dm.data,
            reg=SnitchStreamerMemoryMap.dimension[dim].stride,
            value=op.value,
            comment=f"dm {op.dm.data} dim {dim} stride",
        )
        rewriter.replace_op(
            op,
            [*ops],
            [],
        )


class LowerSsrSetDimensionSourceOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionSourceOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.data
        ops = write_ssr_config_ops(
            dm=op.dm.data,
            reg=SnitchStreamerMemoryMap.dimension[dim].source,
            value=op.value,
            comment=f"dm {op.dm.data} dim {dim} source",
        )
        rewriter.replace_op(
            op,
            [*ops],
            [],
        )


class LowerSsrSetDimensionDestinationOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetDimensionDestinationOp, rewriter: PatternRewriter, /
    ):
        dim: int = op.dimension.data
        ops = write_ssr_config_ops(
            dm=op.dm.data,
            reg=SnitchStreamerMemoryMap.dimension[dim].destination,
            value=op.value,
            comment=f"dm {op.dm.data} dim {dim} destination",
        )
        rewriter.replace_op(
            op,
            [*ops],
            [],
        )


class LowerSsrSetStreamRepetitionOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch.SsrSetStreamRepetitionOp, rewriter: PatternRewriter, /
    ):
        ops = write_ssr_config_ops(
            dm=op.dm.data,
            reg=SnitchStreamerMemoryMap.repeat,
            value=op.value,
            comment=f"dm {op.dm.data} repeat",
        )
        rewriter.replace_op(
            op,
            [*ops],
            [],
        )


class LowerSsrEnable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrEnableOp, rewriter: PatternRewriter, /):
        get_stream_ops = tuple(riscv_snitch.GetStreamOp(res.type) for res in op.results)
        rewriter.replace_op(
            op,
            [
                riscv.CsrrsiOp(
                    csr=IntegerAttr(SnitchStreamerMemoryMap.csr, i32),
                    immediate=IntegerAttr(1, i32),
                    rd=riscv.Registers.ZERO,
                    comment="SSR enable",
                ),
                *get_stream_ops,
            ],
            tuple(op.stream for op in get_stream_ops),
        )


class LowerSsrDisable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snitch.SsrDisableOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            [
                riscv.CsrrciOp(
                    csr=IntegerAttr(SnitchStreamerMemoryMap.csr, i32),
                    immediate=IntegerAttr(1, i32),
                    rd=riscv.Registers.ZERO,
                    comment="SSR disable",
                )
            ],
            [],
        )


@dataclass(frozen=True)
class LowerSnitchPass(ModulePass):
    name = "lower-snitch"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
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
