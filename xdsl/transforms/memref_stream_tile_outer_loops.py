from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from xdsl.context import MLContext
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
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
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
    val: SSAValue,
    m: AffineMapAttr,
    apply_operands: Sequence[SSAValue],
    upper_bounds: Sequence[int],
    location: InsertPoint,
):

    name_hint_prefix = val.name_hint + "_" if val.name_hint is not None else ""
    apply_ops = tuple(
        affine.ApplyOp(
            apply_operands,
            AffineMapAttr(AffineMap(m.data.num_dims, m.data.num_symbols, (res,))),
        )
        for res in m.data.results
    )
    offset_vals = tuple(apply_op.result for apply_op in apply_ops)
    for offset_val in offset_vals:
        offset_val.name_hint = name_hint_prefix + "offset"
    Rewriter.insert_op(apply_ops, location)

    # New subview shape
    max_values = tuple(ub - 1 for ub in upper_bounds)
    sizes = tuple(
        apply_op.map.data.eval(max_values, ())[0] + 1 for apply_op in apply_ops
    )

    source_type = val.type
    if not isinstance(source_type, memref.MemRefType):
        raise DiagnosticException("Cannot create subview from non-memref type")
    source_type = cast(MemRefType[Attribute], source_type)
    layout_attr = source_type.layout
    strides = tuple(source_type.get_strides())
    if isinstance(layout_attr, NoneAttr):
        layout_attr = StridedLayoutAttr(strides, None)
        dest_type = MemRefType(source_type.element_type, sizes, layout_attr)
    else:
        dest_type = source_type

    # While not technically incorrect, the new size should be smaller than the source,
    # instead of just reusing the source shape.
    # We could compute the new shape and make it dynamic but it's not strictly
    # speaking necessar for the rest of the pipeline.
    subview_op = memref.Subview.get(
        val,
        offset_vals,
        sizes,
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
    Replaces a given generic op with a for loop containing an op with the ub at the
    specified index set to 1.
    """
    if (
        generic_op.iterator_types.data[index].data
        != memref_stream.IteratorType.PARALLEL
    ):
        raise DiagnosticException("Cannot ")

    ops: list[Operation] = [
        zero_op := arith.Constant(IntegerAttr.from_index_int_value(0)),
        one_op := arith.Constant(IntegerAttr.from_index_int_value(1)),
        ub_op := arith.Constant(generic_op.bounds.data[index]),
    ]
    zero_val = zero_op.result
    zero_val.name_hint = "c0"
    one_op.result.name_hint = "c1"
    ub_op.result.name_hint = "ub"

    for_block = Block((yield_op := scf.Yield(),), arg_types=(IndexType(),))

    loc = InsertPoint.before(yield_op)

    ops.append(scf.For(zero_op, ub_op, one_op, (), Region(for_block)))

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
        insert_subview(input_val, m, input_apply_operands, input_upper_bounds, loc)
        for input_val, m in zip(
            generic_op.inputs, generic_op.indexing_maps.data[:num_inputs], strict=True
        )
    )
    new_outputs = tuple(
        insert_subview(output_val, m, output_apply_operands, output_upper_bounds, loc)
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
    )

    Rewriter.insert_op(new_generic_op, loc)

    rewriter.replace_matched_op(ops)

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
class MemrefStreamTileOuterLoopsPass(ModulePass):
    """
    Materializes loops around memref_stream.generic operations that have greater than
    specified number of non-1 upper bounds.
    """

    name = "memref-stream-tile-outer-loops"

    target_rank: int = field()

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            TileGenericPattern(self.target_rank),
            apply_recursively=False,
        ).rewrite_module(op)
