from typing import cast

from xdsl.dialects import (
    builtin,
    riscv,
    riscv_snitch,
    snitch,
    snitch_stream,
    stream,
)
from xdsl.ir import MLContext, Operation
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
        # https://github.com/pulp-platform/snitch/blob/d026f47843f0ea6c269244c4e6851e0e09141ec3/sw/snRuntime/src/ssr.h#L73
        #
        # 4d loop reproduced here:
        #
        # // Configure an SSR data mover for a 4D loop nest.
        # // b0: Inner-most bound (limit of loop)
        # // b3: Outer-most bound (limit of loop)
        # // s0: increment size of inner-most loop
        # inline void snrt_ssr_loop_4d(enum snrt_ssr_dm dm, size_t b0, size_t b1,
        #                              size_t b2, size_t b3, size_t s0, size_t s1,
        #                              size_t s2, size_t s3) {
        #     --b0;
        #     --b1;
        #     --b2;
        #     --b3;
        #     write_ssr_cfg(REG_BOUNDS + 0, dm, b0);
        #     write_ssr_cfg(REG_BOUNDS + 1, dm, b1);
        #     write_ssr_cfg(REG_BOUNDS + 2, dm, b2);
        #     write_ssr_cfg(REG_BOUNDS + 3, dm, b3);
        #     size_t a = 0;
        #     write_ssr_cfg(REG_STRIDES + 0, dm, s0 - a);
        #     a += s0 * b0;
        #     write_ssr_cfg(REG_STRIDES + 1, dm, s1 - a);
        #     a += s1 * b1;
        #     write_ssr_cfg(REG_STRIDES + 2, dm, s2 - a);
        #     a += s2 * b2;
        #     write_ssr_cfg(REG_STRIDES + 3, dm, s3 - a);
        #     a += s3 * b3;
        # }

        rank = len(op.ub)
        if rank > 4:
            raise NotImplementedError(
                "Only 1d, 2d, 3d, or 4d loop stride patterns are supported"
            )

        ints = tuple(builtin.IntAttr(i) for i in range(rank))

        b_ops = tuple(riscv.LiOp(b.data) for b in op.ub.data)
        s_ops = tuple(riscv.LiOp(s.data) for s in op.strides.data)
        new_b_ops = tuple(riscv.AddiOp(b_op.rd, -1) for b_op in b_ops)
        set_bound_ops = tuple(
            snitch.SsrSetDimensionBoundOp(new_b_op, op.dm, i)
            for (i, new_b_op) in zip(ints, new_b_ops)
        )

        new_ops: list[Operation] = [
            *b_ops,
            *s_ops,
            *new_b_ops,
            *set_bound_ops,
            snitch.SsrSetDimensionStrideOp(s_ops[0], op.dm, ints[0]),
            a_op := riscv.LiOp(0, rd=riscv.IntRegisterType.unallocated()),
        ]

        for i in range(1, rank):
            a_inc_op = riscv.MulOp(
                new_b_ops[i - 1], s_ops[i - 1], rd=riscv.IntRegisterType.unallocated()
            )
            new_a_op = riscv.AddOp(
                a_op, a_inc_op, rd=riscv.IntRegisterType.unallocated()
            )
            stride_op = riscv.SubOp(
                s_ops[i], new_a_op, rd=riscv.IntRegisterType.unallocated()
            )
            set_stride_op = snitch.SsrSetDimensionStrideOp(stride_op.rd, op.dm, ints[i])
            new_ops.extend((a_inc_op, new_a_op, stride_op, set_stride_op))
            a_op = new_a_op

        rewriter.insert_op_before_matched_op(new_ops)
        rewriter.erase_matched_op()


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
        pattern_type = cast(snitch_stream.StridePatternType, op.pattern.type)

        rewriter.replace_matched_op(
            (
                snitch.SsrSetDimensionSourceOp(
                    op.pointer, op.dm, builtin.IntAttr(pattern_type.data - 1)
                ),
                riscv_snitch.GetStreamOp(stream_type),
            )
        )


class LowerStridedWriteOp(RewritePattern):
    """
    Lower a strided write to snrt_ssr_write.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StridedWriteOp, rewriter: PatternRewriter, /
    ):
        pattern_type = cast(snitch_stream.StridePatternType, op.pattern.type)

        rewriter.replace_matched_op(
            (
                snitch.SsrSetDimensionDestinationOp(
                    op.pointer,
                    op.dm,
                    builtin.IntAttr(pattern_type.data - 1),
                ),
                riscv_snitch.GetStreamOp(op.stream.type),
            )
        )


class LowerStreamingRegionOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StreamingRegionOp, rewriter: PatternRewriter, /
    ):
        # Create strided reads and writes
        # Insert stream begin
        # Inline body
        # Insert stream end
        input_count = len(op.inputs)
        output_count = len(op.outputs)
        stream_count = input_count + output_count

        # If there is a single pattern specified, then it should be set for all streams
        if len(op.stride_patterns) == 1:
            pattern = op.stride_patterns[0]
            patterns = (pattern,) * stream_count
        else:
            patterns = op.stride_patterns

        dms = tuple(range(stream_count))

        strided_read_ops = tuple(
            snitch_stream.StridedReadOp(
                input,
                pattern,
                riscv.Registers.FT[index],
                dm=builtin.IntAttr(dm),
            )
            for index, (input, pattern, dm) in enumerate(
                zip(op.inputs, patterns[:input_count], dms[:input_count], strict=True)
            )
        )

        rewriter.insert_op_before_matched_op(strided_read_ops)

        strided_write_ops = tuple(
            snitch_stream.StridedWriteOp(
                output,
                pattern,
                riscv.Registers.FT[index + input_count],
                dm=builtin.IntAttr(dm),
            )
            for index, (output, pattern, dm) in enumerate(
                zip(op.outputs, patterns[input_count:], dms[input_count:], strict=True)
            )
        )
        rewriter.insert_op_before_matched_op(strided_write_ops)

        rewriter.insert_op_before_matched_op(snitch.SsrEnable())

        block = op.body.block

        for i, arg in zip(strided_read_ops + strided_write_ops, block.args):
            arg.replace_by(i.stream)

        for arg in reversed(block.args):
            rewriter.erase_block_argument(arg)

        rewriter.inline_block_before_matched_op(block)

        rewriter.replace_matched_op(snitch.SsrDisable())


class ConvertSnitchStreamToSnitch(ModulePass):
    name = "convert-snitch-stream-to-snitch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # StridedWrite and StridePattern ops are rewritten to remove their results, so we
        # have to first lower the ops that use the results in `stream`, and then the ops
        # themselves.
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerStreamingRegionOp(),
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                    LowerStridePatternOp(),
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)
