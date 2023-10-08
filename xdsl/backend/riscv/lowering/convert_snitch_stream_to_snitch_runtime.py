from math import prod
from typing import cast

from xdsl.dialects import builtin, riscv, snitch_runtime, snitch_stream, stream
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerStridedReadOp(RewritePattern):
    """
    Lower a strided read to snrt_ssr_loop_2d followed immediately by snrt_ssr_read.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StridedReadOp, rewriter: PatternRewriter, /
    ):
        stream_type = cast(
            stream.InputStreamType[riscv.FloatRegisterType], op.stream.type
        )
        dm_index = riscv.Registers.FT.index(stream_type.element_type)

        assert len(op.ub) == 2

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(dm_index),
                b0 := riscv.LiOp(op.ub.data[0].data),
                b1 := riscv.LiOp(op.ub.data[1].data),
                s0 := riscv.LiOp(op.strides.data[0].data),
                s1 := riscv.LiOp(op.strides.data[1].data),
                dim_2d := riscv.LiOp(1),
                snitch_runtime.SsrLoop2dOp(dm, (b0, b1), (s0, s1)),
                snitch_runtime.SsrReadOp(dm, dim_2d, op.pointer),
                riscv.GetFloatRegisterOp(stream_type.element_type),
            ]
        )


class LowerStridedWriteOp(RewritePattern):
    """
    Lower a strided write to snrt_ssr_loop_2d followed immediately by snrt_ssr_write.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StridedWriteOp, rewriter: PatternRewriter, /
    ):
        stream_type = cast(
            stream.InputStreamType[riscv.FloatRegisterType], op.stream.type
        )
        dm_index = riscv.Registers.FT.index(stream_type.element_type)

        assert len(op.ub) == 2

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(dm_index),
                b0 := riscv.LiOp(op.ub.data[0].data),
                b1 := riscv.LiOp(op.ub.data[1].data),
                s0 := riscv.LiOp(op.strides.data[0].data),
                s1 := riscv.LiOp(op.strides.data[1].data),
                dim_2d := riscv.LiOp(1),
                snitch_runtime.SsrLoop2dOp(dm, (b0, b1), (s0, s1)),
                snitch_runtime.SsrWriteOp(dm, dim_2d, op.pointer),
                riscv.GetFloatRegisterOp(stream_type.element_type),
            ]
        )


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.GenericOp, rewriter: PatternRewriter, /
    ):
        rewriter.insert_op_before_matched_op(snitch_runtime.SsrEnableOp())

        block = op.body.block

        for i, arg in zip(op.inputs, block.args):
            arg.replace_by(i)

        for arg in reversed(block.args):
            rewriter.erase_block_argument(arg)

        loop_count = riscv.LiOp(prod(r.data for r in op.static_loop_ranges) - 1)
        rewriter.insert_op_before_matched_op(loop_count)
        rewriter.replace_matched_op(
            [
                riscv.FrepOuter(
                    loop_count,
                    rewriter.move_region_contents_to_new_regions(op.body),
                    builtin.IntAttr(0),
                    builtin.IntAttr(0),
                ),
                snitch_runtime.SsrDisableOp(),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.YieldOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(riscv.FrepYieldOp(*op.operands))


class ConvertSnitchStreamToSnitchRuntime(ModulePass):
    name = "convert-snitch-stream-to-snitch-runtime"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                    LowerGenericOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
