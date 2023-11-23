"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""

import operator
from collections.abc import Sequence
from functools import reduce
from itertools import accumulate
from math import prod

from xdsl.dialects import arith, linalg, stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    ModuleOp,
    ShapedType,
)
from xdsl.ir import MLContext
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def offset_map_from_shape(shape: Sequence[int]) -> AffineMap:
    """
    Given a list of lengths for each dimension of a memref, and the number of bytes per
    element, returns the map from indices to an offset in bytes in memory. The resulting
    map has one result expression.

    e.g.:
    ```
    my_list = [1, 2, 3, 4, 5, 6]
    shape = [2, 3]
    for i in range(2):
        for j in range(3):
            k = i * 3 + j
            el = my_list[k]
            print(el) # -> 1, 2, 3, 4, 5, 6

    map = offset_map_from_strides([3, 1])

    for i in range(2):
        for j in range(3):
            k = map.eval(i, j)
            el = my_list[k]
            print(el) # -> 1, 2, 3, 4, 5, 6
    ```
    """
    if not shape:
        # Return empty map to avoid reducing over an empty sequence
        return AffineMap(0, 0, (AffineExpr.constant(1),))

    strides: tuple[int, ...] = tuple(
        accumulate(reversed(shape), operator.mul, initial=1)
    )[:-1]

    return AffineMap(
        len(shape),
        0,
        (
            reduce(
                operator.add,
                (
                    AffineExpr.dimension(i) * stride
                    for i, stride in enumerate(reversed(strides))
                ),
            ),
        ),
    )


def strides_for_affine_map(affine_map: AffineMap, shape: Sequence[int]) -> list[int]:
    if affine_map.num_symbols:
        raise ValueError("Cannot create strides for affine map with symbols")
    offset_map = offset_map_from_shape(shape)
    composed = offset_map.compose(affine_map)

    zeros = [0] * composed.num_dims

    result: list[int] = []

    for i in range(composed.num_dims):
        zeros[i] = 1
        result.append(composed.eval(zeros, ())[0])
        zeros[i] = 0

    return result


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        if op.res:
            # Cannot lower linalg generic op with results
            return

        if not all(i.data for i in op.iterator_types):
            # Cannot lower linalg generic with non-parallel iterator types
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
        repeat_count = prod(ub)
        ub_attr = ArrayAttr(IntAttr(b) for b in ub)

        first_affine_map = op.indexing_maps.data[0]
        all_maps_same = not any(
            other != first_affine_map for other in op.indexing_maps.data[1:]
        )
        input_shapes: tuple[tuple[int, ...], ...] = tuple(
            v.type.get_shape() if isinstance(v.type, ShapedType) else ()
            for v in op.inputs
        )
        output_shapes: tuple[tuple[int, ...], ...] = tuple(
            v.type.get_shape() if isinstance(v.type, ShapedType) else ()
            for v in op.outputs
        )
        operand_shapes = input_shapes + output_shapes
        first_shape = operand_shapes[0]
        all_shapes_same = not any(other != first_shape for other in operand_shapes[1:])

        if all_maps_same and all_shapes_same:
            first_strides = ArrayAttr(
                [
                    IntAttr(stride)
                    for stride in strides_for_affine_map(
                        first_affine_map.data, first_shape
                    )
                ]
            )
            dm_all = IntAttr(31)
            stride_pattern_ops = [
                stream.StridePatternOp(ub_attr, first_strides, dm_all)
            ]
        else:
            stride_pattern_ops = [
                stream.StridePatternOp(
                    ub_attr,
                    ArrayAttr(
                        [
                            IntAttr(stride)
                            for stride in strides_for_affine_map(affine_map.data, shape)
                        ]
                    ),
                    IntAttr(i),
                )
                for i, (affine_map, shape) in enumerate(
                    zip(op.indexing_maps, operand_shapes)
                )
            ]

        rewriter.replace_matched_op(
            [
                *stride_pattern_ops,
                repeat_count := arith.Constant(IntegerAttr(repeat_count, IndexType())),
                stream.GenericOp(
                    repeat_count.result,
                    op.inputs,
                    op.outputs,
                    [p.pattern for p in stride_pattern_ops],
                    rewriter.move_region_contents_to_new_regions(op.body),
                ),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.YieldOp, rewriter: PatternRewriter):
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
