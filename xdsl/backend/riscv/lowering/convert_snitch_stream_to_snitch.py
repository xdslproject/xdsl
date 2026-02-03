from xdsl.context import Context
from xdsl.dialects import (
    builtin,
    riscv,
    snitch,
    snitch_stream,
)
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def insert_stride_pattern_ops(
    rewriter: PatternRewriter,
    target_op: Operation,
    ub: builtin.ArrayAttr[builtin.IntAttr],
    strides: builtin.ArrayAttr[builtin.IntAttr],
    repeat: builtin.IntAttr,
    dm: builtin.IntAttr,
):
    """
    `ub` and `strides` must go from the outermost dimension inwards
    """
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
    #     write_ssr_cfg(REG_BOUNDS + 0, dm, b0);
    #     --b1;
    #     write_ssr_cfg(REG_BOUNDS + 1, dm, b1);
    #     --b2;
    #     write_ssr_cfg(REG_BOUNDS + 2, dm, b2);
    #     --b3;
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

    rank = len(ub)
    if rank > 4:
        raise NotImplementedError(
            "Only 1d, 2d, 3d, or 4d loop stride patterns are supported"
        )

    ints = tuple(builtin.IntAttr(i) for i in range(rank))

    b_ops = tuple(riscv.LiOp(b.data) for b in reversed(ub.data))
    new_b_ops = tuple(riscv.AddiOp(b_op.rd, -1) for b_op in b_ops)
    set_bound_ops = tuple(
        snitch.SsrSetDimensionBoundOp(new_b_op, dm, i)
        for (i, new_b_op) in zip(ints, new_b_ops, strict=True)
    )
    interleaved_b_set_bound_ops = tuple(
        x for t in zip(new_b_ops, set_bound_ops) for x in t
    )
    s_ops = tuple(riscv.LiOp(s.data) for s in reversed(strides.data))

    new_ops: list[Operation] = [
        *b_ops,
        *interleaved_b_set_bound_ops,
        *s_ops,
        snitch.SsrSetDimensionStrideOp(s_ops[0], dm, ints[0]),
        a_op := riscv.LiOp(0),
    ]

    for i in range(1, rank):
        a_inc_op = riscv.MulOp(new_b_ops[i - 1], s_ops[i - 1])
        new_a_op = riscv.AddOp(a_op, a_inc_op)
        stride_op = riscv.SubOp(s_ops[i], new_a_op)
        set_stride_op = snitch.SsrSetDimensionStrideOp(stride_op.rd, dm, ints[i])
        new_ops.extend((a_inc_op, new_a_op, stride_op, set_stride_op))
        a_op = new_a_op

    # Always reset the repetition count, even if it's the default
    new_ops.extend(
        (
            repeat_op := riscv.LiOp(repeat.data - 1),
            snitch.SsrSetStreamRepetitionOp(repeat_op.rd, dm),
        )
    )

    rewriter.insert_op(new_ops, InsertPoint.before(target_op))


class LowerStreamingRegionOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_stream.StreamingRegionOp, rewriter: PatternRewriter, /
    ):
        # Set up stream stride patterns
        # Set up stream pointers
        # Insert stream begin
        # Inline body
        # Insert stream end
        input_count = len(op.inputs)
        output_count = len(op.outputs)
        stream_count = input_count + output_count

        # If there is a single pattern specified, then it should be set for all streams
        if len(op.stride_patterns) == 1 and stream_count != 1:
            pattern = op.stride_patterns.data[0]
            patterns = (pattern,) * stream_count
            # Set same pattern for all data movers
            insert_stride_pattern_ops(
                rewriter,
                op,
                pattern.ub,
                pattern.strides,
                pattern.repeat,
                builtin.IntAttr(31),
            )
        else:
            patterns = op.stride_patterns.data
            # Set separate pattern per data mover
            for dm, pattern in enumerate(patterns):
                insert_stride_pattern_ops(
                    rewriter,
                    op,
                    pattern.ub,
                    pattern.strides,
                    pattern.repeat,
                    builtin.IntAttr(dm),
                )

        dms = tuple(range(stream_count))

        set_source_ops = tuple(
            snitch.SsrSetDimensionSourceOp(
                input,
                dm=builtin.IntAttr(dm),
                dimension=builtin.IntAttr(pattern.rank() - 1),
            )
            for input, pattern, dm in zip(
                op.inputs, patterns[:input_count], dms[:input_count], strict=True
            )
        )

        rewriter.insert_op(set_source_ops)

        set_destination_ops = tuple(
            snitch.SsrSetDimensionDestinationOp(
                output,
                dm=builtin.IntAttr(dm),
                dimension=builtin.IntAttr(pattern.rank() - 1),
            )
            for output, pattern, dm in zip(
                op.outputs, patterns[input_count:], dms[input_count:], strict=True
            )
        )
        rewriter.insert_op(set_destination_ops)

        block = op.body.block

        rewriter.insert_op(enable_op := snitch.SsrEnableOp(block.arg_types))

        for val, arg in zip(enable_op.streams, block.args):
            arg.replace_by(val)

        for arg in reversed(block.args):
            rewriter.erase_block_argument(arg)

        rewriter.inline_block(block, InsertPoint.before(op))

        rewriter.replace_op(op, snitch.SsrDisableOp())


class ConvertSnitchStreamToSnitch(ModulePass):
    name = "convert-snitch-stream-to-snitch"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # StridedWrite and StridePattern ops are rewritten to remove their results, so we
        # have to first lower the ops that use the results in `stream`, and then the ops
        # themselves.
        PatternRewriteWalker(
            LowerStreamingRegionOp(), apply_recursively=False
        ).rewrite_module(op)
