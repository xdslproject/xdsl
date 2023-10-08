import operator
from itertools import accumulate

from xdsl.backend.riscv.lowering.utils import (
    cast_block_args_to_regs,
    cast_operands_to_regs,
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
from xdsl.ir.affine.affine_map import AffineMap
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
        rewriter.replace_matched_op(
            snitch_stream.GenericOp(
                op.inputs,
                op.outputs,
                new_region,
                op.static_loop_ranges,
            )
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


def strides_for_affine_map(
    affine_map: AffineMap, ub: list[int], bitwidth: int
) -> list[int]:
    identity = AffineMap.identity(affine_map.num_dims)
    if affine_map == identity:
        prod_dims: list[int] = list(
            accumulate(reversed(ub), operator.mul, initial=bitwidth)
        )[1::-1]
        return prod_dims
    elif affine_map == identity.transpose:
        prod_dims: list[int] = list(accumulate(ub, operator.mul, initial=bitwidth))[:-1]
        return prod_dims
    else:
        raise NotImplementedError(f"Unsupported affine map {affine_map}")


class LowerStridedReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.StridedReadOp, rewriter: PatternRewriter):
        (pointer,) = cast_operands_to_regs(rewriter)
        strides = strides_for_affine_map(
            op.indexing_map.data, [b.data for b in op.ub], 8
        )

        rewriter.replace_matched_op(
            snitch_stream.StridedReadOp(
                pointer,
                riscv.FloatRegisterType.unallocated(),
                op.ub,
                ArrayAttr([IntAttr(stride) for stride in strides]),
            )
        )


class LowerStridedWriteOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stream.StridedWriteOp, rewriter: PatternRewriter):
        (pointer,) = cast_operands_to_regs(rewriter)
        strides = strides_for_affine_map(
            op.indexing_map.data, [b.data for b in op.ub], 8
        )

        rewriter.replace_matched_op(
            snitch_stream.StridedWriteOp(
                pointer,
                riscv.FloatRegisterType.unallocated(),
                op.ub,
                ArrayAttr([IntAttr(stride) for stride in strides]),
            )
        )


class StreamToSnitchStreamPass(ModulePass):
    name = "stream-to-snitch-stream"

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
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                ]
            )
        ).rewrite_module(op)
