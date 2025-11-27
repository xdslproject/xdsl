from collections.abc import Callable, Sequence
from itertools import compress
from typing import TypeAlias

from xdsl.dialects import affine, arith, scf
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint


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
                # used_dims = affine_map.used_dims_bit_vector()
                used_dims_vector = tuple(
                    dim in used_dims for dim in range(affine_map.num_dims)
                )
                unused_dims_vector = tuple(
                    not used_dim for used_dim in used_dims_vector
                )
                new_index_vals = tuple(compress(new_index_vals, used_dims_vector))
                new_affine_map = new_affine_map.drop_dims(unused_dims_vector)

            rewriter.insert_op(
                apply_op := affine.ApplyOp(
                    new_index_vals,
                    AffineMapAttr(new_affine_map),
                ),
                insertion_point,
            )

            output_indices.append(apply_op.result)

    return output_indices


INSERT_LOAD: TypeAlias = Callable[
    [
        int,
        Sequence[SSAValue],
        PatternRewriter,
        InsertPoint,
    ],
    SSAValue,
]


INSERT_STORE: TypeAlias = Callable[
    [
        int,
        SSAValue,
        Sequence[SSAValue],
        PatternRewriter,
        InsertPoint,
    ],
    Operation,
]


def _insert_loop_nest(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    zero_op: arith.ConstantOp,
    one_op: arith.ConstantOp,
    bounds: Sequence[SSAValue],
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
    loops: list[scf.ForOp] = []
    index = IndexType()

    for i, ub in enumerate(bounds):
        loop = scf.ForOp(
            zero_op.result,
            ub,
            one_op.result,
            iter_args,
            Region(Block(arg_types=(index, *iter_arg_types))),
        )
        iter_args = loop.body.block.args[1:]
        loops.append(loop)
        rewriter.insert_op(loop, insertion_point)
        if i:
            # Do not insert yield outside of outermost loop
            rewriter.insert_op(scf.YieldOp(*loop.results), InsertPoint.after(loop))

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
            rewriter.insert_op(
                scf.YieldOp(*results), InsertPoint.at_end(loop.body.block)
            )

        insertion_point = InsertPoint.at_start(loop.body.block)

    return loops[0].results


def _insert_load_ops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    ind_vars: Sequence[BlockArgument],
    affine_map_attrs: Sequence[AffineMapAttr],
    operands: Sequence[SSAValue],
    args: Sequence[BlockArgument],
    insert_load: INSERT_LOAD,
    index_increment: int = 0,
) -> Sequence[tuple[int, SSAValue]]:
    """
    Inserts the load operations at the specified insertion point.
    The `ind_vars` are the induction variables for this loop nest, corresponding to the
    domain of the affine maps.
    The `operands` are the structures to load from.
    The `args` are the block arguments corresponding to the use of the load; if there are
    no uses, the loads are not inserted.
    The `affine_map_attrs`, `operands`, and `args` must have the same length.
    Returns a tuple of integers indicating the locations of the returned values, and
    the values themselves.
    The integer values are incremented by `index_increment`.
    """
    res: list[tuple[int, SSAValue]] = []
    for i, arg in enumerate(args):
        if not arg.uses:
            continue
        res_val = insert_load(
            i + index_increment,
            ind_vars,
            rewriter,
            insertion_point,
        )
        res.append((i + index_increment, res_val))
    return res


def _insert_store_ops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    ind_vars: Sequence[BlockArgument],
    yield_operands: Sequence[SSAValue],
    insert_store: INSERT_STORE,
):
    """
    Inserts the store operations at the specified insertion point.
    The `ind_vars` are the induction variables for this loop nest, corresponding to the
    domain of the affine maps.
    The `yield_operands` are the operands of the yield operation that is being replaced
    with stores.
    The `output_operands` are the structures to store into.
    The `output_indexing_maps`, `yield_operands`, and `output_operands` must have the same length.
    """
    for i, yield_value in enumerate(yield_operands):
        insert_store(i, yield_value, ind_vars, rewriter, insertion_point)


def rewrite_generic_to_loops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    ubs: Sequence[int],
    load_indexing_maps: Sequence[AffineMapAttr],
    store_indexing_maps: Sequence[AffineMapAttr],
    load_operands: Sequence[SSAValue],
    store_operands: Sequence[SSAValue],
    block: Block,
    insert_load: INSERT_LOAD,
    insert_store: INSERT_STORE,
) -> None:
    # Create loop nest lb (0), step (1), and ubs
    # ubs are calculated from affine maps and memref dimensions

    bound_constant_ops = tuple(
        arith.ConstantOp(IntegerAttr.from_index_int_value(ub)) for ub in ubs
    )
    rewriter.insert_op(bound_constant_ops)
    bound_constant_values = tuple(op.result for op in bound_constant_ops)

    zero_op = arith.ConstantOp(IntegerAttr.from_index_int_value(0))
    one_op = arith.ConstantOp(IntegerAttr.from_index_int_value(1))
    if bound_constant_values:
        rewriter.insert_op((zero_op, one_op))

    def make_body(
        rewriter: PatternRewriter,
        insertion_point: InsertPoint,
        ind_vars: Sequence[BlockArgument],
        iter_args: Sequence[SSAValue],
    ) -> Sequence[SSAValue]:
        assert not iter_args

        loaded_values = _insert_load_ops(
            rewriter,
            insertion_point,
            ind_vars,
            load_indexing_maps,
            load_operands,
            block.args,
            insert_load,
        )

        for i, val in loaded_values:
            block.args[i].replace_by(val)

        yield_op = block.last_op
        assert yield_op is not None

        # Erase the yield op, we still have access to its operands
        rewriter.erase_op(yield_op)

        while block.args:
            rewriter.erase_block_argument(block.args[0])

        rewriter.inline_block(block, insertion_point)

        _insert_store_ops(
            rewriter,
            insertion_point,
            ind_vars,
            yield_op.operands,
            insert_store,
        )

        return ()

    _insert_loop_nest(
        rewriter,
        insertion_point,
        zero_op,
        one_op,
        bound_constant_values,
        (),
        make_body,
    )

    rewriter.erase_op(rewriter.current_operation)


def rewrite_generic_to_imperfect_loops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    outer_ubs: Sequence[int],
    inner_ubs: Sequence[int],
    outer_load_indexing_maps: Sequence[AffineMapAttr],
    inner_load_indexing_maps: Sequence[AffineMapAttr],
    store_indexing_maps: Sequence[AffineMapAttr],
    outer_load_operands: Sequence[SSAValue],
    inner_load_operands: Sequence[SSAValue],
    store_operands: Sequence[SSAValue],
    outer_load_block_args: Sequence[BlockArgument],
    inner_load_block_args: Sequence[BlockArgument],
    block: Block,
    insert_load: INSERT_LOAD,
    insert_store: INSERT_STORE,
) -> None:
    # Create loop nest lb (0), step (1), and ubs
    # ubs are calculated from affine maps and memref dimensions

    outer_bound_constant_ops = tuple(
        arith.ConstantOp(IntegerAttr.from_index_int_value(ub)) for ub in outer_ubs
    )
    inner_bound_constant_ops = tuple(
        arith.ConstantOp(IntegerAttr.from_index_int_value(ub)) for ub in inner_ubs
    )
    rewriter.insert_op(outer_bound_constant_ops, insertion_point)
    rewriter.insert_op(inner_bound_constant_ops, insertion_point)
    outer_bound_constant_values = tuple(op.result for op in outer_bound_constant_ops)
    inner_bound_constant_values = tuple(op.result for op in inner_bound_constant_ops)

    zero_op = arith.ConstantOp(IntegerAttr.from_index_int_value(0))
    one_op = arith.ConstantOp(IntegerAttr.from_index_int_value(1))
    if outer_bound_constant_values or inner_bound_constant_values:
        rewriter.insert_op((zero_op, one_op))

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
            outer_load_indexing_maps,
            outer_load_operands,
            outer_load_block_args,
            insert_load,
            index_increment=len(inner_load_block_args),
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
                inner_load_indexing_maps,
                inner_load_operands,
                inner_load_block_args,
                insert_load,
            )

            # Replace block argument use with iter args
            for (i, _), arg in zip(
                outer_loaded_values,
                inner_iter_args,
                strict=True,
            ):
                block.args[i].replace_by(arg)

            # Replace block argument use with load op results
            for i, val in inner_loaded_values:
                block.args[i].replace_by(val)

            yield_op = block.last_op
            assert yield_op is not None

            # Erase the yield op, we still have access to its operands
            rewriter.erase_op(yield_op)

            # Inline generic body into innermost scf loop
            # The operands have already been remapped

            while block.args:
                rewriter.erase_block_argument(block.args[0])

            rewriter.inline_block(block, insertion_point)

            return yield_op.operands

        # Insert inner loop nest, from the outtermost loop inwards
        inner_loop_nest_results = _insert_loop_nest(
            rewriter,
            insertion_point,
            zero_op,
            one_op,
            inner_bound_constant_values,
            tuple(val for _, val in outer_loaded_values),
            inner_make_body,
        )

        # Finally, add store ops
        _insert_store_ops(
            rewriter,
            insertion_point,
            outer_ind_vars,
            inner_loop_nest_results,
            insert_store,
        )

        return ()

    # Insert outer loop nest, from the outtermost loop inwards
    _insert_loop_nest(
        rewriter,
        insertion_point,
        zero_op,
        one_op,
        outer_bound_constant_values,
        (),
        outer_make_body,
    )

    rewriter.erase_op(rewriter.current_operation)
