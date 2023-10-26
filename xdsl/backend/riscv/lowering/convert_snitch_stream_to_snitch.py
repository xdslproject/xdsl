from typing import cast

from xdsl.dialects import (
    builtin,
    riscv,
    riscv_snitch,
    snitch,
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
        # reference implementation:
        # // Configure an SSR data mover for a 2D loop nest.
        #
        # inline void snrt_ssr_loop_2d(enum snrt_ssr_dm dm, size_t b0, size_t b1,
        #                              size_t s0, size_t s1) {
        #     --b0;
        #     --b1;
        #     write_ssr_cfg(REG_BOUNDS + 0, dm, b0);
        #     write_ssr_cfg(REG_BOUNDS + 1, dm, b1);
        #     size_t a = 0;
        #     write_ssr_cfg(REG_STRIDES + 0, dm, s0 - a);
        #     a += s0 * b0;
        #     write_ssr_cfg(REG_STRIDES + 1, dm, s1 - a);
        # }
        dim = len(op.ub)
        if dim != 2:
            raise NotImplementedError("Only 2d loop stride patterns are supported")

        int_0 = builtin.IntAttr(0)
        int_1 = builtin.IntAttr(1)

        b = tuple(b.data for b in op.ub.data)
        s = tuple(s.data for s in op.strides.data)

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(op.dm.data),
                b0 := riscv.LiOp(b[0]),
                b1 := riscv.LiOp(b[1]),
                s0 := riscv.LiOp(s[0]),
                s1 := riscv.LiOp(s[1]),
                new_b0 := riscv.AddiOp(b0, -1),
                new_b1 := riscv.AddiOp(b1, -1),
                snitch.SsrSetDimensionBoundOp(dm, new_b0, int_0),
                snitch.SsrSetDimensionBoundOp(dm, new_b1, int_1),
                snitch.SsrSetDimensionStrideOp(dm, s0, int_0),
                a0 := riscv.MulOp(new_b0, s0, rd=riscv.IntRegisterType.unallocated()),
                stride_1 := riscv.SubOp(s1, a0, rd=riscv.IntRegisterType.unallocated()),
                snitch.SsrSetDimensionStrideOp(dm, stride_1, int_1),
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
        dim = op.pattern
        assert isinstance(dim.owner, riscv.LiOp)
        dim_v = dim.owner.immediate
        assert isinstance(dim_v, builtin.IntegerAttr)

        rewriter.replace_matched_op(
            [
                dm := riscv.LiOp(op.dm.data),
                snitch.SsrSetDimensionSourceOp(dm, op.pointer, dim_v.value),
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
        dim = op.pattern
        assert isinstance(dim.owner, riscv.LiOp)
        dim_v = dim.owner.immediate
        assert isinstance(dim_v, builtin.IntegerAttr)

        rewriter.insert_op_before_matched_op(
            [
                dm := riscv.LiOp(op.dm.data),
                snitch.SsrSetDimensionDestinationOp(dm, op.pointer, dim_v.value),
            ]
        )

        rewriter.erase_matched_op()


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.GenericOp, rewriter: PatternRewriter, /
    ):
        rewriter.insert_op_before_matched_op(snitch.SsrEnable())

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
                snitch.SsrDisable(),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.YieldOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_matched_op(riscv_snitch.FrepYieldOp(*op.operands))


class ConvertSnitchStreamToSnitch(ModulePass):
    name = "convert-snitch-stream-to-snitch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerStridePatternOp(),
                    LowerGenericOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                ]
            )
        ).rewrite_module(op)
