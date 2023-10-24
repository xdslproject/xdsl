from xdsl.backend.riscv.lowering.utils import (
    cast_block_args_to_regs,
    cast_ops_for_values,
    move_ops_for_value,
    register_type_for_type,
)
from xdsl.dialects import riscv, snitch_stream, stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.GenericOp, rewriter: PatternRewriter):
        # Input streams are currently streams of values, but they will be replaced with
        # streams of registers, so don't need to do anything for operands.
        # The only thing to do is to rewrite the block arguments to be float registers.
        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        cast_block_args_to_regs(new_region.block, rewriter)
        cast_op, repeat_count = cast_ops_for_values((op.repeat_count,))
        rewriter.replace_matched_op(
            [
                *cast_op,
                snitch_stream.GenericOp(
                    repeat_count[0],
                    op.inputs,
                    op.outputs,
                    new_region,
                ),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.YieldOp, rewriter: PatternRewriter):
        new_ops = list[Operation]()
        new_operands = list[SSAValue]()

        b = op.parent_block()

        for operand in rewriter.current_operation.operands:
            new_type = register_type_for_type(operand.type)
            cast_op = UnrealizedConversionCastOp.get(
                (operand,), (new_type.unallocated(),)
            )
            new_ops.append(cast_op)
            new_operand = cast_op.results[0]
            if operand.owner is b:
                # Have to move from input register to output register
                mv_op, new_operand = move_ops_for_value(
                    new_operand, new_type.unallocated()
                )
                new_ops.append(mv_op)

            new_operands.append(new_operand)

        rewriter.insert_op_before_matched_op(new_ops)

        rewriter.replace_matched_op(snitch_stream.YieldOp(*new_operands))


class LowerStridePatternOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.StridePatternOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            snitch_stream.StridePatternOp(
                op.ub,
                ArrayAttr(IntAttr(stride.data * 8) for stride in op.strides),
                op.dm,
            )
        )


class LowerStridedReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.StridedReadOp, rewriter: PatternRewriter):
        new_ops, new_operands = cast_ops_for_values((op.memref,))
        rewriter.insert_op_before_matched_op(new_ops)

        (pointer,) = new_operands

        rewriter.replace_matched_op(
            snitch_stream.StridedReadOp(
                pointer,
                op.pattern,
                riscv.FloatRegisterType.unallocated(),
                op.dm,
            )
        )


class LowerStridedWriteOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.StridedWriteOp, rewriter: PatternRewriter):
        new_ops, new_operands = cast_ops_for_values((op.memref,))
        rewriter.insert_op_before_matched_op(new_ops)

        (pointer,) = new_operands

        rewriter.replace_matched_op(
            snitch_stream.StridedWriteOp(
                pointer,
                op.pattern,
                riscv.FloatRegisterType.unallocated(),
                op.dm,
            )
        )


class ConvertStreamToSnitchStreamPass(ModulePass):
    name = "convert-stream-to-snitch-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # Lower yield before generic
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    LowerStridePatternOp(),
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                ]
            )
        ).rewrite_module(op)
