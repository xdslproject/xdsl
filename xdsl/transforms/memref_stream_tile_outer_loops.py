from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from xdsl.context import Context
from xdsl.dialects import affine, arith, memref, memref_stream, scf
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import DiagnosticException


def insert_subview(
    memref_val: SSAValue,
    affine_map: AffineMap,
    dim_offsets: Sequence[SSAValue],
    upper_bounds: Sequence[int],
    location: InsertPoint,
):
    """
    A helper method to insert a subview from the input `memref_val` to one with new
    upper bounds, given that it will be accessed with the specified affine map.
    `dim_offsets` are the operands to use to determine the new offset, and
    `upper_bounds` the new shape.
    Any new operations should be inserted at `location`.
    """
    name_hint_prefix = (
        memref_val.name_hint + "_" if memref_val.name_hint is not None else ""
    )
    apply_ops = tuple(
        affine.ApplyOp(
            dim_offsets,
            AffineMapAttr(
                AffineMap(affine_map.num_dims, affine_map.num_symbols, (res,))
            ),
        )
        for res in affine_map.results
    )
    offset_vals = tuple(apply_op.result for apply_op in apply_ops)
    for offset_val in offset_vals:
        offset_val.name_hint = name_hint_prefix + "offset"
    Rewriter.insert_op(apply_ops, location)

    # The new upper bounds are determined in the following way:
    # 1. Take the maximum values each individual index could take (ub - 1)
    # 2. Apply the access pattern, yielding the new maximum value
    #   (This assumes that the affine expression is always increasing in the inputs)
    # 3. Add 1 to yield new upper bound
    source_max_indices = tuple(ub - 1 for ub in upper_bounds)
    dest_max_indices = affine_map.eval(source_max_indices, ())
    dest_shape = tuple(max_index + 1 for max_index in dest_max_indices)

    source_type = memref_val.type
    if not isinstance(source_type, memref.MemRefType):
        raise DiagnosticException("Cannot create subview from non-memref type")
    source_type = cast(MemRefType, source_type)
    layout_attr = source_type.layout
    assert (strides := source_type.get_strides())
    strides = tuple(strides)
    match layout_attr:
        case NoneAttr():
            layout_attr = StridedLayoutAttr(strides, None)
            dest_type = MemRefType(source_type.element_type, dest_shape, layout_attr)
        case StridedLayoutAttr():
            # We currently only support subviews from memref with statically known strides and dynamic offsets
            if any(stride is None for stride in layout_attr.get_strides()):
                raise DiagnosticException(
                    f"Layout attr for tiling {layout_attr} not yet supported"
                )
            if layout_attr.get_offset() is not None:
                raise DiagnosticException(
                    f"Layout attr for tiling {layout_attr} not yet supported"
                )
            dest_type = source_type
        case _:
            raise DiagnosticException(
                f"Unsupported layout attr for tiling {layout_attr}"
            )

    subview_op = memref.SubviewOp.get(
        memref_val,
        offset_vals,
        dest_shape,
        (1,) * len(strides),
        dest_type,
    )
    subview_op.result.name_hint = name_hint_prefix + "subview"
    Rewriter.insert_op(subview_op, location)
    return subview_op.result


def materialize_loop(
    rewriter: PatternRewriter, generic_op: memref_stream.GenericOp, index: int
) -> Sequence[Operation]:
    """
    Replaces a given generic op with a for loop containing an op with the upper bound at
    the specified index set to 1.
    """
    if (
        generic_op.iterator_types.data[index].data
        != memref_stream.IteratorType.PARALLEL
    ):
        raise DiagnosticException(
            "Cannot materialize a loop for a non-parallel iterator"
        )

    ops: list[Operation] = [
        zero_op := arith.ConstantOp(IntegerAttr.from_index_int_value(0)),
        one_op := arith.ConstantOp(IntegerAttr.from_index_int_value(1)),
        ub_op := arith.ConstantOp(generic_op.bounds.data[index]),
    ]
    zero_val = zero_op.result
    zero_val.name_hint = "c0"
    one_op.result.name_hint = "c1"
    ub_op.result.name_hint = "ub"

    for_block = Block((yield_op := scf.YieldOp(),), arg_types=(IndexType(),))

    loc = InsertPoint.before(yield_op)

    ops.append(scf.ForOp(zero_op, ub_op, one_op, (), Region(for_block)))

    index_val = for_block.args[0]
    index_val.name_hint = "i"

    input_apply_operands: list[SSAValue] = [zero_val] * len(generic_op.iterator_types)
    input_apply_operands[index] = index_val
    output_apply_operands: list[SSAValue] = [zero_val] * (
        len(generic_op.iterator_types)
        - sum(
            it.data == memref_stream.IteratorType.REDUCTION
            for it in generic_op.iterator_types
        )
    )
    output_apply_operands[index] = index_val

    input_upper_bounds = list(ub.value.data for ub in generic_op.bounds)
    input_upper_bounds[index] = 1
    output_upper_bounds = list(
        ub.value.data
        for (it, ub) in zip(generic_op.iterator_types, generic_op.bounds)
        if it.data != memref_stream.IteratorType.REDUCTION
    )
    output_upper_bounds[index] = 1

    num_inputs = len(generic_op.inputs)

    new_inputs = tuple(
        insert_subview(input_val, m.data, input_apply_operands, input_upper_bounds, loc)
        for input_val, m in zip(
            generic_op.inputs, generic_op.indexing_maps.data[:num_inputs], strict=True
        )
    )
    new_outputs = tuple(
        insert_subview(
            output_val, m.data, output_apply_operands, output_upper_bounds, loc
        )
        for output_val, m in zip(
            generic_op.outputs, generic_op.indexing_maps.data[num_inputs:], strict=True
        )
    )

    new_bounds = list(generic_op.bounds)
    new_bounds[index] = IntegerAttr.from_index_int_value(1)

    new_generic_op = memref_stream.GenericOp(
        new_inputs,
        new_outputs,
        generic_op.inits,
        Rewriter.move_region_contents_to_new_regions(generic_op.body),
        generic_op.indexing_maps,
        generic_op.iterator_types,
        ArrayAttr(new_bounds),
        generic_op.init_indices,
        generic_op.doc,
        generic_op.library_call,
    )

    Rewriter.insert_op(new_generic_op, loc)

    rewriter.replace_op(generic_op, ops)

    return ops


@dataclass(frozen=True)
class TileGenericPattern(RewritePattern):
    target_rank: int = field()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        ubs = tuple(bound.value.data for bound in op.bounds)
        effective_rank = sum(ub > 1 for ub in ubs)
        if effective_rank <= self.target_rank:
            return
        index = 0
        for index, ub in enumerate(ubs):
            if ub > 1:
                break
        materialize_loop(rewriter, op, index)


@dataclass(frozen=True)
class MemRefStreamTileOuterLoopsPass(ModulePass):
    """
    Materializes loops around memref_stream.generic operations that have greater than
    specified number of non-1 upper bounds.
    """

    name = "memref-stream-tile-outer-loops"

    target_rank: int = field()

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            TileGenericPattern(self.target_rank),
        ).rewrite_module(op)
