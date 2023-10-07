import operator
from itertools import accumulate

from xdsl.backend.riscv.lowering.utils import (
    cast_block_args_to_regs,
    cast_operands_to_regs,
)
from xdsl.dialects import riscv, snitch_stream, stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    ModuleOp,
)
from xdsl.ir import MLContext
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
        rewriter.replace_matched_op(
            snitch_stream.YieldOp(*cast_operands_to_regs(rewriter))
        )


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
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    LowerYieldOp(),
                    LowerStridedReadOp(),
                    LowerStridedWriteOp(),
                ]
            )
        ).rewrite_module(op)
