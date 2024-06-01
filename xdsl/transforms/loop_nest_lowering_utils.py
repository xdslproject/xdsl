from collections.abc import Callable, Sequence
from itertools import compress

from xdsl.builder import InsertPoint
from xdsl.dialects import affine, arith, linalg, memref_stream, scf
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr
from xdsl.ir import Block, BlockArgument, Operation, OpResult, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.pattern_rewriter import PatternRewriter


def indices_for_map(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
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

            rewriter.insert_op_at_location(
                apply_op := affine.ApplyOp(
                    new_index_vals,
                    AffineMapAttr(new_affine_map),
                ),
                insertion_point,
            )

            output_indices.append(apply_op.result)

    return output_indices


def _insert_loop_nest(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    zero_op: arith.Constant,
    one_op: arith.Constant,
    bounds: tuple[OpResult, ...],
    iter_args: Sequence[SSAValue],
    make_body: Callable[
        [PatternRewriter, InsertPoint, Sequence[BlockArgument], Sequence[SSAValue]],
        Sequence[SSAValue],
    ],
) -> Sequence[SSAValue]:
    """
    Creates a perfect loop nest, populating the innermost body with the provided
    `make_body` function.
    If `iter_args` are passed in, the loop nest will pass them from parent loop to child
    loop, and the results of `make_body` are expected to be equal in length to
    `iter_args`.

    `zero_op` and `one_op` are operations ƒor the `0` and `1` index constants for the
    loop nest lower bound and step. The upper bounds are specified by the `bounds`
    arguement.
    """
    if not bounds:
        return make_body(rewriter, insertion_point, (), iter_args)

    iter_arg_types = tuple(arg.type for arg in iter_args)
    loops: list[scf.For] = []
    index = IndexType()

    for i, ub in enumerate(bounds):
        loop = scf.For(
            zero_op.result,
            ub,
            one_op.result,
            iter_args,
            Region(Block(arg_types=(index, *iter_arg_types))),
        )
        iter_args = loop.body.block.args[1:]
        loops.append(loop)
        rewriter.insert_op_at_location(loop, insertion_point)
        results = loop.results

        if i + 1 == len(bounds):
            # Innermost loop iteration
            results = make_body(
                rewriter,
                InsertPoint.at_start(loop.body.block),
                tuple(loop.body.block.args[0] for loop in loops),
                iter_args,
            )
            if len(results) != len(iter_args):
                raise ValueError(
                    "Unexpected number of results from `make_body` helper "
                    f"({len(results)}), expected {len(iter_args)}"
                )
        rewriter.insert_op_at_end(scf.Yield(*results), loop.body.block)
        insertion_point = InsertPoint.at_start(loop.body.block)

    return loops[0].results


def _insert_load_ops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    ind_vars: Sequence[BlockArgument],
    affine_map_attrs: Sequence[AffineMapAttr],
    operands: Sequence[SSAValue],
    args: Sequence[BlockArgument],
    load: Callable[
        [SSAValue, Sequence[SSAValue], PatternRewriter, InsertPoint], SSAValue
    ],
) -> Sequence[tuple[int, SSAValue]]:
    res: list[tuple[int, SSAValue]] = []
    for i, (affine_map_attr, operand, arg) in enumerate(
        zip(affine_map_attrs, operands, args, strict=True)
    ):
        if not arg.uses:
            continue
        affine_map = affine_map_attr.data
        indices = indices_for_map(rewriter, insertion_point, affine_map, ind_vars)
        res_val = load(operand, indices, rewriter, insertion_point)
        res.append((i, res_val))
    return res


def _insert_store_ops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    ind_vars: Sequence[BlockArgument],
    output_indexing_maps: Sequence[AffineMapAttr],
    yield_operands: Sequence[SSAValue],
    output_operands: Sequence[SSAValue],
    store: Callable[[SSAValue, SSAValue, Sequence[SSAValue]], Operation],
):
    for affine_map_attr, yield_value, ref in zip(
        output_indexing_maps, yield_operands, output_operands, strict=True
    ):
        affine_map = affine_map_attr.data
        indices = indices_for_map(rewriter, insertion_point, affine_map, ind_vars)
        store_op = store(yield_value, ref, indices)
        rewriter.insert_op_at_location(store_op, insertion_point)


def rewrite_generic_to_loops(
    rewriter: PatternRewriter,
    op: linalg.Generic | memref_stream.GenericOp,
    load: Callable[
        [SSAValue, Sequence[SSAValue], PatternRewriter, InsertPoint], SSAValue
    ],
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

    ins_count = len(op.inputs)
    outer_bound_count = min(
        len(bound_constant_values), *(m.data.num_dims for m in op.indexing_maps)
    )
    outer_bounds = bound_constant_values[:outer_bound_count]
    inner_bounds = bound_constant_values[outer_bound_count:]
    if outer_bound_count != len(bound_constant_values):
        # Imperfectly nested
        inner_load_count = ins_count
        outer_indexing_maps = op.indexing_maps.data[ins_count:]
        inner_indexing_maps = op.indexing_maps.data[:ins_count]
        outer_op_block_args = op.body.block.args[ins_count:]
        inner_op_block_args = op.body.block.args[:ins_count]
        outer_load_operands = op.outputs
        inner_load_operands = op.inputs

    else:
        inner_load_count = 0
        outer_indexing_maps = op.indexing_maps.data
        inner_indexing_maps = ()
        outer_op_block_args = op.body.block.args
        inner_op_block_args = ()
        outer_load_operands = tuple(op.operands)
        inner_load_operands = ()

    def outer_make_body(
        rewriter: PatternRewriter,
        insertion_point: InsertPoint,
        outer_ind_vars: Sequence[BlockArgument],
        outer_iter_args: Sequence[SSAValue],
    ) -> Sequence[SSAValue]:
        assert not outer_iter_args

        # Add load ops
        outer_loaded_values = _insert_load_ops(
            rewriter,
            insertion_point,
            outer_ind_vars,
            outer_indexing_maps,
            outer_load_operands,
            outer_op_block_args,
            load,
        )

        def inner_make_body(
            rewriter: PatternRewriter,
            insertion_point: InsertPoint,
            inner_ind_vars: Sequence[BlockArgument],
            inner_iter_args: Sequence[SSAValue],
        ):
            # Add load ops
            inner_loaded_values = _insert_load_ops(
                rewriter,
                insertion_point,
                (*outer_ind_vars, *inner_ind_vars),
                inner_indexing_maps,
                inner_load_operands,
                inner_op_block_args,
                load,
            )

            # Replace block argument use with iter args
            for (i, _), arg in zip(
                outer_loaded_values,
                inner_iter_args,
                strict=True,
            ):
                op.body.block.args[i + inner_load_count].replace_by(arg)

            # Replace block argument use with load op results
            for i, val in inner_loaded_values:
                op.body.block.args[i].replace_by(val)

            yield_op = op.body.block.last_op
            assert isinstance(yield_op, linalg.YieldOp | memref_stream.YieldOp)

            # Erase the yield op, we still have access to its operands
            rewriter.erase_op(yield_op)

            # Inline generic body into innermost scf loop
            # The operands have already been remapped

            while op.body.block.args:
                rewriter.erase_block_argument(op.body.block.args[0])

            rewriter.inline_block_at_location(op.body.block, insertion_point)

            return yield_op.operands

        # Insert inner loop nest, from the outtermost loop inwards
        inner_loop_nest_results = _insert_loop_nest(
            rewriter,
            insertion_point,
            zero_op,
            one_op,
            inner_bounds,
            tuple(val for _, val in outer_loaded_values),
            inner_make_body,
        )

        # Finally, add store ops
        output_indexing_maps = op.indexing_maps.data[-len(op.outputs) :]
        output_operands = op.operands[-len(op.outputs) :]
        _insert_store_ops(
            rewriter,
            insertion_point,
            outer_ind_vars,
            output_indexing_maps,
            inner_loop_nest_results,
            output_operands,
            store,
        )

        return ()

    # Insert outer loop nest, from the outtermost loop inwards
    _insert_loop_nest(
        rewriter,
        InsertPoint.before(op),
        zero_op,
        one_op,
        outer_bounds,
        (),
        outer_make_body,
    )

    rewriter.erase_matched_op()
