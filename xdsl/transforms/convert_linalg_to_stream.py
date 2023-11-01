"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""

import operator
from itertools import accumulate
from math import prod

from xdsl.dialects import arith, linalg, stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    ModuleOp,
)
from xdsl.ir import MLContext
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def strides_for_affine_map(
    affine_map: AffineMap, ub: list[int], bitwidth: int
) -> list[int]:
    if affine_map == AffineMap.identity(affine_map.num_dims):
        prod_dims: list[int] = list(
            accumulate(reversed(ub), operator.mul, initial=bitwidth)
        )[1::-1]
        return prod_dims
    elif affine_map == AffineMap.transpose_map():
        prod_dims: list[int] = list(accumulate(ub, operator.mul, initial=bitwidth))[:-1]
        return prod_dims
    else:
        raise NotImplementedError(f"Unsupported affine map {affine_map}")


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        if op.res:
            # Cannot lower linalg generic op with results
            return

        # Streams are exclusively readable or writable
        # Look at block arguments in body, if an output argument is used, bail, otherwise remove it.

        block = op.body.block
        input_count = len(op.inputs)
        output_args = block.args[input_count:]
        if any(o.uses for o in output_args):
            # Cannot lower inout args to stream
            return

        for o in output_args:
            rewriter.erase_block_argument(o)

        ub = op.get_static_loop_ranges()
        rank = len(ub)
        repeat_count = prod(ub)
        ub_attr = ArrayAttr(IntAttr(b) for b in ub)

        first_affine_map = op.indexing_maps.data[0]
        all_maps_same = not any(
            other != first_affine_map for other in op.indexing_maps.data[1:]
        )

        if all_maps_same:
            first_strides = ArrayAttr(
                [
                    IntAttr(stride)
                    for stride in strides_for_affine_map(
                        first_affine_map.data, [b.data for b in ub_attr], 1
                    )
                ]
            )
            dm_all = IntAttr(31)
            stride_pattern_ops = [
                stream.StridePatternOp(ub_attr, first_strides, dm_all)
            ]
            stride_patterns = stride_pattern_ops * len(ub)
        else:
            stride_pattern_ops = [
                stream.StridePatternOp(
                    ub_attr,
                    ArrayAttr(
                        [
                            IntAttr(stride)
                            for stride in strides_for_affine_map(
                                affine_map.data, [b.data for b in ub_attr], 1
                            )
                        ]
                    ),
                    IntAttr(i),
                )
                for i, affine_map in enumerate(op.indexing_maps)
            ]
            stride_patterns = stride_pattern_ops

        new_inputs = [
            stream.StridedReadOp(
                memref,
                stride_pattern.pattern,
                IntAttr(i),
                IntAttr(rank),
            )
            for i, (memref, stride_pattern) in enumerate(
                zip(op.inputs, stride_patterns)
            )
        ]
        new_outputs = [
            stream.StridedWriteOp(
                memref,
                stride_pattern.pattern,
                IntAttr(i + len(op.inputs)),
                IntAttr(rank),
            )
            for i, (memref, stride_pattern) in enumerate(
                zip(op.outputs, stride_patterns[-len(op.outputs) :])
            )
        ]

        rewriter.replace_matched_op(
            [
                *stride_pattern_ops,
                *new_inputs,
                *new_outputs,
                repeat_count := arith.Constant(IntegerAttr(repeat_count, IndexType())),
                stream.GenericOp(
                    repeat_count.result,
                    [i.stream for i in new_inputs],
                    [o.stream for o in new_outputs],
                    rewriter.move_region_contents_to_new_regions(op.body),
                ),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Yield, rewriter: PatternRewriter):
        rewriter.replace_matched_op(stream.YieldOp(*op.operands))


class ConvertLinalgToStreamPass(ModulePass):
    name = "convert-linalg-to-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
