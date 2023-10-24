from typing import cast

from xdsl.dialects import (
    builtin,
    riscv,
    riscv_snitch,
    snitch_runtime,
    snitch_stream,
    stream,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerStridePatternOp(RewritePattern):
    """
    Lower a stride pattern to snrt_ssr_loop_2d.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StridePatternOp, rewriter: PatternRewriter, /
    ):
        dim = len(op.ub)
        if dim != 2:
            raise NotImplementedError("Only 2d loop stride patterns are supported")

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(op.dm.data),
                b0 := riscv.LiOp(op.ub.data[0].data),
                b1 := riscv.LiOp(op.ub.data[1].data),
                s0 := riscv.LiOp(op.strides.data[0].data),
                s1 := riscv.LiOp(op.strides.data[1].data),
                snitch_runtime.SsrLoop2dOp(dm, (b0, b1), (s0, s1)),
                # The result is rewritten to be the dimensionality of the stream
                # configuration, which is `dim-1`.
                riscv.LiOp(dim - 1),
            ]
        )


class LowerStridedReadOp(RewritePattern):
    """
    Lower a strided read to snrt_ssr_read.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StridedReadOp, rewriter: PatternRewriter, /
    ):
        stream_type = cast(
            stream.ReadableStreamType[riscv.FloatRegisterType], op.stream.type
        )

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(op.dm.data),
                snitch_runtime.SsrReadOp(dm, op.pattern, op.pointer),
                riscv.GetFloatRegisterOp(stream_type.element_type),
            ]
        )


class LowerStridedWriteOp(RewritePattern):
    """
    Lower a strided write to snrt_ssr_write.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StridedWriteOp, rewriter: PatternRewriter, /
    ):
        stream_type = cast(
            stream.ReadableStreamType[riscv.FloatRegisterType], op.stream.type
        )

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(op.dm.data),
                snitch_runtime.SsrWriteOp(dm, op.pattern, op.pointer),
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

        loop_count = riscv.AddiOp(op.repeat_count, -1)
        rewriter.insert_op_before_matched_op(loop_count)
        rewriter.replace_matched_op(
            [
                riscv_snitch.FrepOuter(
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
        rewriter.replace_matched_op(riscv_snitch.FrepYieldOp(*op.operands))


class ConvertSnitchStreamToSnitchRuntime(ModulePass):
    name = "convert-snitch-stream-to-snitch-runtime"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerStridePatternOp(),
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                    LowerGenericOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
