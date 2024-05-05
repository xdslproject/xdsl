from collections.abc import Callable, Sequence
from itertools import compress

from xdsl.dialects import affine, arith, linalg, memref_stream, scf
from xdsl.dialects.builtin import (
    AffineMapAttr,
    IndexType,
    IntegerAttr,
)
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.pattern_rewriter import (
    PatternRewriter,
)


def indices_for_map(
    rewriter: PatternRewriter,
    target_op: Operation,
    affine_map: AffineMap,
    input_index_vals: Sequence[SSAValue],
) -> Sequence[SSAValue]:
    """
    Given an affine map mapping iteration indices to indices to a memref, return the
    indices into the corresponding memref. The number of returned SSA values corresponds
    to the number of results of the affine map. If the result is an affine dimension
    expression, then return the corresponding input index. Otherwise, add an
    `affine.apply` operation that calculates the indices, reducing the expression to only
    the relevant dimensions.

    For example, the map `(d0, d1, d2, d3) -> (d0 + d2)` when applied to indices
    `(a, b, c, d)` is transformed to the map `(d0, d1) -> (d0 + d1)` when applied to
    indices `(a, c)`.

    The `affine.apply` operations are inserted before `target_op`.
    """
    if affine_map.num_symbols:
        raise NotImplementedError("Cannot create indices for affine map with symbols")
    output_indices: list[SSAValue] = []
    for expr in affine_map.results:
        if isinstance(expr, AffineDimExpr):
            output_indices.append(input_index_vals[expr.position])
        else:
            used_dims = expr.used_dims()
            new_index_vals = input_index_vals
            new_affine_map = AffineMap(
                affine_map.num_dims, affine_map.num_symbols, (expr,)
            )
            if len(used_dims) != affine_map.num_dims:
                # Remove unused dims
                selectors = tuple(
                    dim in used_dims for dim in range(affine_map.num_dims)
                )
                new_index_vals = tuple(compress(new_index_vals, selectors))
                new_affine_map = new_affine_map.compress_dims(selectors)

            rewriter.insert_op_before(
                apply_op := affine.ApplyOp(
                    new_index_vals,
                    AffineMapAttr(new_affine_map),
                ),
                target_op,
            )

            output_indices.append(apply_op.result)

    return output_indices


def rewrite_generic_to_loops(
    rewriter: PatternRewriter,
    op: linalg.Generic | memref_stream.GenericOp,
    load: Callable[[SSAValue, Sequence[SSAValue]], Operation],
    store: Callable[[SSAValue, SSAValue, Sequence[SSAValue]], Operation],
) -> None:
    # Create loop nest lb (0), step (1), and ubs
    # ubs are calculated from affine maps and memref dimensions

    ubs = op.get_static_loop_ranges()

    bound_constant_ops = tuple(
        arith.Constant(IntegerAttr.from_index_int_value(ub)) for ub in ubs
    )
    rewriter.insert_op_before_matched_op(bound_constant_ops)
    bound_constant_values = tuple(op.result for op in bound_constant_ops)

    zero_op = arith.Constant(IntegerAttr.from_index_int_value(0))
    one_op = arith.Constant(IntegerAttr.from_index_int_value(1))
    if bound_constant_values:
        rewriter.insert_op_before_matched_op((zero_op, one_op))

    index = IndexType()

    # Insert loop nest, from the outtermost loop inwards

    loop_args: list[BlockArgument] = []
    insertion_target: Operation = op

    for ub in bound_constant_values:
        loop = scf.For(
            zero_op.result,
            ub,
            one_op.result,
            (),
            Region(Block((yield_op := scf.Yield(),), arg_types=(index,))),
        )
        loop_args.append(loop.body.block.args[0])
        rewriter.insert_op_before(loop, insertion_target)
        insertion_target = yield_op

    # Add load ops before the innermost scf.yield operation

    for affine_map_attr, operand, arg in zip(
        op.indexing_maps.data, op.operands, op.body.block.args, strict=True
    ):
        if not arg.uses:
            continue
        affine_map = affine_map_attr.data
        indices = indices_for_map(rewriter, insertion_target, affine_map, loop_args)
        load_op = load(operand, indices)
        rewriter.insert_op_before(load_op, insertion_target)
        arg.replace_by(load_op.results[0])

    # Add store ops before the yield operation in the generic body

    yield_op = op.body.block.last_op
    assert isinstance(yield_op, linalg.YieldOp | memref_stream.YieldOp)

    output_indexing_maps = op.indexing_maps.data[-len(op.outputs) :]
    output_operands = op.operands[-len(op.outputs) :]
    for affine_map_attr, yield_value, ref in zip(
        output_indexing_maps, yield_op.operands, output_operands, strict=True
    ):
        affine_map = affine_map_attr.data
        indices = indices_for_map(rewriter, insertion_target, affine_map, loop_args)
        store_op = store(yield_value, ref, indices)
        rewriter.insert_op_before(store_op, yield_op)

    # Now that the linalg yield op operands have been converted to stores, remove

    rewriter.erase_op(yield_op)

    # Inline generic body into innermost scf loop
    # The operands have already been remapped

    while op.body.block.args:
        rewriter.erase_block_argument(op.body.block.args[0])

    rewriter.inline_block_before(op.body.block, insertion_target)

    # Erase generic

    rewriter.erase_matched_op()
