from xdsl.backend.riscv.lowering.utils import (
    cast_ops_for_values,
    move_ops_for_value,
    register_type_for_type,
)
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import (
    builtin,
    memref_stream,
    riscv,
    riscv_snitch,
    snitch_stream,
    stream,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Block, MLContext, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerGenericOp(RewritePattern):
    """
    Rewrites stream.generic to be a streaming region with an frep, but defers substituting block arguments to stream.yield lowering.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref_stream.GenericOp, rewriter: PatternRewriter):
        # Cast inputs to RISC-V registers
        rep_count_cast_ops, new_rep_counts = cast_ops_for_values((op.repeat_count,))
        (new_rep_count,) = new_rep_counts
        input_cast_ops, new_inputs = cast_ops_for_values(op.inputs)
        output_cast_ops, new_outputs = cast_ops_for_values(op.outputs)

        # Replace by
        # 1. streaming region
        # 2. block arguments are streams
        # 3. explicit loop inside

        input_stream_types = (
            stream.ReadableStreamType(riscv.FloatRegisterType.unallocated()),
        ) * len(new_inputs)
        output_stream_types = (
            stream.WritableStreamType(riscv.FloatRegisterType.unallocated()),
        ) * len(new_outputs)

        streaming_region_body = Region(
            Block(arg_types=input_stream_types + output_stream_types)
        )

        streaming_region = snitch_stream.StreamingRegionOp(
            new_inputs, new_outputs, op.stride_patterns, streaming_region_body
        )

        with ImplicitBuilder(streaming_region_body):
            rep_count_minus_one = riscv.AddiOp(new_rep_count, -1).rd
            frep = riscv_snitch.FrepOuter(
                rep_count_minus_one,
                rewriter.move_region_contents_to_new_regions(op.body),
            )
            with ImplicitBuilder(frep.body):
                riscv_snitch.FrepYieldOp()

        rewriter.replace_matched_op(
            [
                *rep_count_cast_ops,
                *input_cast_ops,
                *output_cast_ops,
                streaming_region,
            ]
        )


class LowerYieldOp(RewritePattern):
    """
    Values yielded in stream.generic are written to streams in snitch_stream. Inputs that do not correspond to a yielded value are converted to stream reads.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref_stream.YieldOp, rewriter: PatternRewriter):
        loop_block = op.parent_block()

        if loop_block is None:
            return

        streaming_region_block = loop_block.parent_block()

        if streaming_region_block is None:
            return

        all_streams = streaming_region_block.args

        output_stream_count = len(op.operands)

        input_streams = all_streams[:-output_stream_count]
        output_streams = all_streams[-output_stream_count:]

        new_ops = list[Operation]()

        # Convert yielded values to stream writes
        for output_stream, operand in zip(output_streams, op.operands, strict=True):
            new_type = register_type_for_type(operand.type)
            cast_op = UnrealizedConversionCastOp.get(
                (operand,), (new_type.unallocated(),)
            )
            new_ops.append(cast_op)
            new_operand = cast_op.results[0]
            if operand.owner is loop_block:
                # Have to move from input register to output register
                mv_op, new_operand = move_ops_for_value(
                    new_operand, new_type.unallocated()
                )
                new_ops.append(mv_op)
            new_ops.append(riscv_snitch.WriteOp(new_operand, output_stream))

        # Convert input values to stream reads
        for input_stream, operand in zip(
            reversed(input_streams), reversed(loop_block.args)
        ):
            rewriter.insert_op_at_start(
                (
                    read_op := riscv_snitch.ReadOp(input_stream),
                    cast_op := builtin.UnrealizedConversionCastOp(
                        operands=[read_op.res], result_types=[operand.type]
                    ),
                ),
                loop_block,
            )
            operand.replace_by(cast_op.results[0])

        # delete block arguments
        for arg in loop_block.args[::-1]:
            rewriter.erase_block_argument(arg)

        rewriter.replace_matched_op(new_ops)


class LowerStridePatternOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.StridePatternOp, rewriter: PatternRewriter
    ):
        rewriter.replace_matched_op(
            snitch_stream.StridePatternOp(
                op.ub,
                ArrayAttr(IntAttr(stride.data * 8) for stride in op.strides),
                op.dm,
            )
        )


class ConvertMemrefStreamToSnitchStreamPass(ModulePass):
    name = "convert-memref-stream-to-snitch-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    LowerStridePatternOp(),
                ]
            )
        ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
