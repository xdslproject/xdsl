from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import assert_never

from xdsl.dialects import affine, arith, linalg, memref, scf, tensor
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    TensorType,
)
from xdsl.dialects.utils import split_dynamic_index_list
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.ir.affine import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException
from xdsl.utils.hints import isa

# iv + index, the index a tiled iteration would have had before it was tiled.
_INDEX_OFFSET_MAP = AffineMapAttr(
    AffineMap(2, 0, (AffineExpr.dimension(0) + AffineExpr.dimension(1),))
)

# min(tile, ub - iv), the size of a tile that may run past the end of its loop.
_PARTIAL_TILE_MIN_MAP = AffineMapAttr(
    AffineMap(
        3,
        0,
        (
            AffineExpr.dimension(0),
            AffineExpr.dimension(1) - AffineExpr.dimension(2),
        ),
    )
)


def _is_zero(tile_size: SSAValue | int) -> bool:
    """Whether a tile size is known to be zero, which means not tiling at all."""
    return isinstance(tile_size, int) and tile_size == 0


def _divides(tile_size: SSAValue | int, loop_range: int) -> bool:
    """
    Whether a tile size is known to divide a loop range, so that every tile of
    that dimension is whole. Neither a size nor a range that is only known once
    the op runs can be shown to.
    """
    return (
        isinstance(tile_size, int) and loop_range >= 0 and loop_range % tile_size == 0
    )


@dataclass(frozen=True)
class OperandTileInfo:
    """
    This records how one operand should be sliced when we enter a tile.
    - `source_type` keeps the original type.
    - `loop_dims` the loop dimension each indexing-map result reads, where it
      reads exactly one, and `None` where it reads an expression over several or
      none, which no single loop range can be read back from.
    """

    source_type: MemRefType[Attribute] | TensorType[Attribute]
    loop_dims: tuple[int | None, ...]

    @staticmethod
    def analyze(
        indexing_map: AffineMap,
        source_type: MemRefType[Attribute] | TensorType[Attribute],
    ) -> "OperandTileInfo":
        """
        Analyze how one operand should be sliced for each tile.
        """

        loop_dims = tuple(
            expr.position if isinstance(expr, AffineDimExpr) else None
            for expr in indexing_map.results
        )
        return OperandTileInfo(source_type, loop_dims)


@dataclass(frozen=True)
class TilingPlan:
    """
    This stores the information needed to turn one op into tiled loop and tiled subview.
    - `loop_ranges` are original static loop ranges.
    - `tiled_dims` the dimensions that really get tiled.
    - `partial_tiled_dims` the tiled dimensions whose loop range is not divisible
      by the tile size, so that their last tile is smaller than the rest.
    - `operand_infos` stores one `OperandTileInfo` per operand.
    - `tile_sizes` are the normalized tile sizes, padded to match the op loop
      count. A tile size that is not known until the op runs is a value.
    """

    loop_ranges: tuple[int, ...]
    tiled_dims: tuple[int, ...]
    partial_tiled_dims: frozenset[int]
    operand_infos: tuple[OperandTileInfo, ...]
    tile_sizes: tuple[SSAValue | int, ...]

    @staticmethod
    def analyze(
        op: linalg.abstract_ops.LinalgStructuredOperation,
        tile_sizes: Sequence[SSAValue | int],
    ) -> "TilingPlan":
        """
        Analyze one supported structured linalg op and return a `TilingPlan`.
        """

        num_loops = op.get_num_loops()
        normalized_tile_sizes: tuple[SSAValue | int, ...] = tuple(
            tile_sizes[:num_loops]
        ) + (0,) * (num_loops - len(tile_sizes))

        # Tiling by zero means leaving a dimension alone, so a dimension is tiled
        # unless its tile size is known to be zero. One that is not known until
        # the op runs cannot be, so it is tiled.
        tiled_dims = tuple(
            dim
            for dim, tile_size in enumerate(normalized_tile_sizes)
            if not _is_zero(tile_size)
        )

        if not tiled_dims:
            return TilingPlan(
                loop_ranges=(),
                tiled_dims=(),
                partial_tiled_dims=frozenset(),
                operand_infos=(),
                tile_sizes=normalized_tile_sizes,
            )

        loop_ranges = _verify_is_tileable(
            op,
            normalized_tile_sizes,
            tiled_dims,
        )

        # A range that is not known until the op runs cannot be shown to divide
        # by its tile size, so it is treated as leaving a leftover tile.
        partial_tiled_dims = frozenset(
            dim
            for dim in tiled_dims
            if not _divides(normalized_tile_sizes[dim], loop_ranges[dim])
        )

        operand_infos_list: list[OperandTileInfo] = []
        for operand, indexing_map in zip(
            op.operands, op.get_indexing_maps(), strict=True
        ):
            source_type = operand.type
            assert isa(source_type, MemRefType | TensorType)
            operand_infos_list.append(
                OperandTileInfo.analyze(indexing_map.data, source_type)
            )
        operand_infos = tuple(operand_infos_list)

        return TilingPlan(
            loop_ranges=loop_ranges,
            tiled_dims=tiled_dims,
            partial_tiled_dims=partial_tiled_dims,
            operand_infos=operand_infos,
            tile_sizes=normalized_tile_sizes,
        )


def _verify_is_tileable(
    op: linalg.abstract_ops.LinalgStructuredOperation,
    tile_sizes: Sequence[SSAValue | int],
    tiled_dims: Sequence[int],
) -> tuple[int, ...]:
    """
    Check whether a structured linalg op is safe to tile.
    """

    if any(
        isinstance(tile_sizes[dim], int) and cast(int, tile_sizes[dim]) < 0
        for dim in tiled_dims
    ):
        raise ValueError("negative tile sizes are not supported")

    # Operands that mix the two are rejected outright rather than tiled, since
    # each kind is written back a different way. MLIR does not consider such an
    # op valid either, requiring pure tensor or pure buffer semantics.
    operand_types = tuple(operand.type for operand in op.operands)
    if any(isa(operand_type, MemRefType) for operand_type in operand_types) and any(
        isa(operand_type, TensorType) for operand_type in operand_types
    ):
        raise ValueError(
            "tiling a linalg op with a mix of memref and tensor operands is "
            "not supported"
        )

    # A reduction dimension is tiled like any other. It is absent from the output
    # indexing maps, being the dimension reduced away, so the output is not sliced
    # along it and each tile reads the value the last one left, accumulating into
    # it. The tiles run in the order the untiled dimension did, which leaves the
    # reduction associating as it did before, so this holds for a reduction that
    # cannot be reassociated, such as one over floats.

    for operand in op.operands:
        raw_operand_type = operand.type

        if not isa(raw_operand_type, MemRefType) and not isa(
            raw_operand_type, TensorType
        ):
            raise NotImplementedError(
                "tiling a linalg op with operands that are neither memrefs nor "
                "tensors is not supported yet"
            )

    loop_ranges = op.get_static_loop_ranges()

    # The bounds of the `scf.for` we build come from `loop_ranges`. One that is
    # dynamic is read at runtime instead, with a `memref.dim` or `tensor.dim` on
    # an operand, so we need an indexing map result that is exactly `dN` to tell
    # us which operand to read and which of its dimensions. A result like
    # `d0 + d1` names no single dimension to read, so a dynamic loop reachable
    # only that way is rejected.
    indexing_maps = tuple(attr.data for attr in op.get_indexing_maps())
    readable_dims = {
        expr.position
        for indexing_map in indexing_maps
        for expr in indexing_map.results
        if isinstance(expr, AffineDimExpr)
    }
    for dim in tiled_dims:
        if loop_ranges[dim] < 0 and dim not in readable_dims:
            raise ValueError(
                "tiling a linalg op dimension whose range is not known until "
                "it runs and that no indexing map reads on its own is not "
                "supported yet"
            )

    return loop_ranges


def _loop_range_sources(
    op: linalg.abstract_ops.LinalgStructuredOperation, plan: TilingPlan
) -> dict[int, tuple[SSAValue, int]]:
    """
    Say where the range of each tiled loop can be read from when it is not known
    until the op runs.

    A loop range comes from the operands it indexes, so a range that is not
    static is read back off one of them, as the operand and the position within
    it that the loop dimension addresses.
    """

    sources: dict[int, tuple[SSAValue, int]] = {}
    for operand, operand_info in zip(op.operands, plan.operand_infos, strict=True):
        for result_index, loop_dim in enumerate(operand_info.loop_dims):
            if loop_dim not in plan.tiled_dims or loop_dim in sources:
                continue
            if plan.loop_ranges[loop_dim] < 0:
                sources[loop_dim] = (operand, result_index)
    return sources


def _build_loop_range(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    source: SSAValue,
    position: int,
) -> SSAValue:
    """
    Read one dimension of an operand, for a loop range only known at runtime.
    """

    position_op = arith.ConstantOp(IntegerAttr(position, IndexType()))
    rewriter.insert(position_op, insertion_point)

    source_type = source.type
    assert isa(source_type, MemRefType | TensorType)
    match source_type:
        case MemRefType():
            dim_op = memref.DimOp(source, position_op)
        case TensorType():
            dim_op = tensor.DimOp(source, position_op)
        case _:
            assert_never(source_type)

    rewriter.insert(dim_op, insertion_point)
    return dim_op.result


def _build_tile_loops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    loop_ranges: Sequence[int],
    tile_sizes: Sequence[SSAValue | int],
    tiled_dims: Sequence[int],
    range_sources: dict[int, tuple[SSAValue, int]],
    iter_args: Sequence[SSAValue] = (),
) -> tuple[list[scf.ForOp], dict[int, SSAValue], InsertPoint]:
    """
    Build the outer tiled loops.

    `iter_args` are threaded through the nest as loop-carried values, as needed
    when tiling operands with value semantics. The outermost loop initialises
    them from `iter_args` itself, and every nested loop initialises them from the
    enclosing loop's block arguments, so the innermost body sees the values
    accumulated by the surrounding iterations. Tiling memrefs carries nothing,
    which leaves the loops without block arguments or results.

    Return:
        - `loops`: the outer `scf.for` ops, outermost first
        - `tiled_loop_ivs`: a map from loop dimensions to induction variables
        - `current_insertion_point`: the place to insert `tiled subview` and the tiled op
    """

    index = IndexType()
    zero = arith.ConstantOp(IntegerAttr(0, index))
    rewriter.insert(zero, insertion_point)

    ubs: dict[int, SSAValue] = {}
    for dim in tiled_dims:
        if loop_ranges[dim] < 0:
            source, position = range_sources[dim]
            ubs[dim] = _build_loop_range(rewriter, insertion_point, source, position)
        else:
            ub_op = arith.ConstantOp(IntegerAttr(loop_ranges[dim], index))
            rewriter.insert(ub_op, insertion_point)
            ubs[dim] = ub_op.result

    steps: dict[int, SSAValue] = {}
    for dim in tiled_dims:
        tile_size = tile_sizes[dim]
        if isinstance(tile_size, SSAValue):
            steps[dim] = tile_size
        else:
            tile_op = arith.ConstantOp(IntegerAttr(tile_size, index))
            rewriter.insert(tile_op, insertion_point)
            steps[dim] = tile_op.result

    current_insertion_point = insertion_point
    loops: list[scf.ForOp] = []
    tiled_loop_ivs: dict[int, SSAValue] = {}
    carried_types = tuple(value.type for value in iter_args)
    carried: Sequence[SSAValue] = iter_args
    for dim in tiled_dims:
        loop = scf.ForOp(
            zero.result,
            ubs[dim],
            steps[dim],
            carried,
            Region(Block(arg_types=(index, *carried_types))),
        )
        rewriter.insert(loop, current_insertion_point)
        loops.append(loop)
        tiled_loop_ivs[dim] = loop.body.block.args[0]
        carried = loop.body.block.args[1:]
        current_insertion_point = InsertPoint.at_start(loop.body.block)

    return loops, tiled_loop_ivs, current_insertion_point


def _build_effective_tile_sizes(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    plan: TilingPlan,
    loops: Sequence[scf.ForOp],
) -> dict[int, SSAValue | int]:
    """
    Give the size of the current tile for each tiled dimension.

    A dimension whose loop range divides by its tile size always gets a whole
    tile, so its size stays the tile size. One that does not has a smaller tile
    on its last iteration, so its size becomes `min(tile, ub - iv)`, which is
    the tile size everywhere except at the end.
    """

    effective: dict[int, SSAValue | int] = {}
    for dim, loop in zip(plan.tiled_dims, loops, strict=True):
        if dim not in plan.partial_tiled_dims:
            effective[dim] = plan.tile_sizes[dim]
            continue

        min_op = affine.MinOp(
            (loop.step, loop.ub, loop.body.block.args[0]), _PARTIAL_TILE_MIN_MAP
        )
        rewriter.insert(min_op, insertion_point)
        effective[dim] = min_op.result

    return effective


def _apply_affine_expr(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    expr: AffineExpr,
    values: Sequence[SSAValue | int],
) -> SSAValue | int:
    """
    Evaluate one result of an indexing map, given a value per loop dimension.

    Returns either an `int`, when the answer is known here, or an SSA value.

    An `affine.apply` is only built when one is needed. A result that is just
    `dN` is the value given for that dimension, and one whose values are all
    ints is worked out here. Otherwise the int values are written into the map
    as constants, and only the SSA ones are passed to the op as operands.
    """

    if isinstance(expr, AffineDimExpr):
        return values[expr.position]

    if all(isinstance(value, int) for value in values):
        return expr.eval(cast(Sequence[int], values), ())

    replacements: list[AffineExpr] = []
    operands: list[SSAValue] = []
    for value in values:
        if isinstance(value, int):
            replacements.append(AffineExpr.constant(value))
        else:
            replacements.append(AffineExpr.dimension(len(operands)))
            operands.append(value)

    substituted = expr.replace_dims_and_symbols(replacements, ())

    # Writing the known values in can leave a map with nothing left to compute,
    # so return the value directly rather than emitting an `affine.apply` that
    # only forwards one operand or yields a constant.
    if isinstance(substituted, AffineConstantExpr):
        return substituted.value
    if isinstance(substituted, AffineDimExpr):
        return operands[substituted.position]

    apply_op = affine.ApplyOp(
        operands,
        AffineMapAttr(AffineMap(len(operands), 0, (substituted,))),
    )
    rewriter.insert(apply_op, insertion_point)
    return apply_op.result


def _decrement(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    value: SSAValue | int,
) -> SSAValue | int:
    """
    Turn a count of elements into the index of its last one, `n` into `n - 1`,
    building an `affine.apply` when the count is not known here.
    """

    return _apply_affine_expr(
        rewriter, insertion_point, AffineExpr.dimension(0) - 1, (value,)
    )


@dataclass(frozen=True)
class SliceParameters:
    """
    Where one operand's tile sits within that operand.

    This is the geometry of the tile, which is the same whether the operand is a
    memref or a tensor, and so does not depend on which op ends up materializing
    the slice.
    """

    offsets: tuple[SSAValue | int, ...]
    sizes: tuple[SSAValue | int, ...]
    strides: tuple[SSAValue | int, ...]

    @staticmethod
    def compute(
        rewriter: PatternRewriter,
        insertion_point: InsertPoint,
        indexing_map: AffineMap,
        operand_info: OperandTileInfo,
        tiled_loop_ivs: dict[int, SSAValue],
        effective_tile_sizes: dict[int, SSAValue | int],
        loop_ranges: Sequence[int],
    ) -> "SliceParameters":
        """
        Compute the offsets and sizes for one operand's `memref.subview` or
        `tensor.extract_slice`.

        There is one offset and one size per operand dimension, and the indexing
        map has one result per operand dimension saying how the loops reach it.

        - A result that no tiled loop appears in is not sliced at all: offset 0,
          and the whole dimension for its size.
        - A result that is just `dN`, for a tiled loop, takes that loop's
          induction variable as its offset and its tile size as its size.
        - Any other result, `d0 + d1` or `d0 * 2`, gets both from an
          `affine.apply`. Its offset evaluates the result with each tiled loop at
          its induction variable and the rest at zero, less what it evaluates to
          with every loop at zero. Its size evaluates it at the last index each
          loop reaches inside the tile, plus one to turn that index back into a
          count.
        """

        source_shape = operand_info.source_type.get_shape()
        num_loops = indexing_map.num_dims

        # The induction variable of each tiled loop, and zero for the rest.
        starts: tuple[SSAValue | int, ...] = tuple(
            tiled_loop_ivs.get(dim, 0) for dim in range(num_loops)
        )
        zeros = (0,) * num_loops
        # The last index each loop reaches inside the tile. This costs an
        # `affine.apply` per loop, so it is built once, and only if some result
        # turns out to need it.
        extents: tuple[SSAValue | int, ...] | None = None

        offsets: list[SSAValue | int] = []
        sizes: list[SSAValue | int] = []
        for result_index, expr in enumerate(indexing_map.results):
            if not (expr.used_dims() & tiled_loop_ivs.keys()):
                offsets.append(0)
                sizes.append(source_shape[result_index])
                continue

            # `dN` on its own: the slice starts at that loop's induction
            # variable and is one tile long. The general case below computes
            # exactly this, so taking it here just avoids emitting an
            # `affine.apply` to say so.
            if isinstance(expr, AffineDimExpr):
                offsets.append(tiled_loop_ivs[expr.position])
                sizes.append(effective_tile_sizes[expr.position])
                continue

            if extents is None:
                extents = tuple(
                    _decrement(
                        rewriter,
                        insertion_point,
                        effective_tile_sizes.get(dim, size),
                    )
                    for dim, size in enumerate(loop_ranges[:num_loops])
                )

            # Take off what the result evaluates to with every loop at zero, so
            # that a constant the map adds does not shift the slice: the tiled op
            # applies the same map inside the slice and adds it back there.
            # Taking it off the expression rather than off the value it produces
            # keeps this to a single `affine.apply`.
            at_zero = expr.eval(zeros, ())
            offsets.append(
                _apply_affine_expr(
                    rewriter,
                    insertion_point,
                    expr if at_zero == 0 else expr - at_zero,
                    starts,
                )
            )
            sizes.append(
                _apply_affine_expr(rewriter, insertion_point, expr + 1, extents)
            )

        return SliceParameters(tuple(offsets), tuple(sizes), (1,) * len(source_shape))


def _build_tiled_slice(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    operand: SSAValue,
    source_type: MemRefType[Attribute] | TensorType[Attribute],
    parameters: SliceParameters,
) -> SSAValue:
    """
    Build the slice of `operand` at the current tile position.

    Memrefs are sliced with a `memref.subview`, which views the source memory,
    and tensors with a `tensor.extract_slice`, which produces a new value. Only
    the op that materializes the slice differs.

    `operand` is the value to slice, which is not always the operand the tile
    info was analyzed from: an output tensor is sliced from the value carried by
    the enclosing loops rather than from the original.
    """

    offsets = parameters.offsets
    sizes = parameters.sizes
    strides = parameters.strides
    try:
        match source_type:
            case MemRefType():
                slice_op = memref.SubviewOp.get(
                    operand,
                    offsets,
                    sizes,
                    strides,
                    memref.SubviewOp.infer_result_type(
                        source_type,
                        offsets,
                        sizes,
                        strides,
                    ),
                )
            case TensorType():
                slice_op = tensor.ExtractSliceOp(
                    operand,
                    offsets,
                    sizes,
                    strides,
                    tensor.ExtractSliceOp.infer_result_type(source_type, sizes),
                )
            case _:
                assert_never(source_type)
    except ValueError as e:
        raise PassFailedException(str(e)) from e

    rewriter.insert(slice_op, insertion_point)

    return slice_op.result


def _build_tiled_insert(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    tiled_value: SSAValue,
    destination: SSAValue,
    parameters: SliceParameters,
) -> SSAValue:
    """
    Write one computed tile back into `destination`, giving the updated tensor.

    Tensors have value semantics, so a tile cannot be written through its slice
    the way a `memref.subview` writes through to the memory it views. The
    parameters are the ones the tile was extracted with, so that the tile is
    written back exactly where it came from.
    """

    static_offsets, offsets = split_dynamic_index_list(
        parameters.offsets, DYNAMIC_INDEX
    )
    static_sizes, sizes = split_dynamic_index_list(parameters.sizes, DYNAMIC_INDEX)
    static_strides, strides = split_dynamic_index_list(
        parameters.strides, DYNAMIC_INDEX
    )

    insert_op = tensor.InsertSliceOp.get(
        source=tiled_value,
        dest=destination,
        static_sizes=static_sizes,
        static_offsets=static_offsets,
        static_strides=static_strides,
        offsets=offsets,
        sizes=sizes,
        strides=strides,
        result_type=destination.type,
    )
    rewriter.insert(insert_op, insertion_point)

    return insert_op.result


def _offset_tiled_indices(
    rewriter: PatternRewriter,
    tiled_op: linalg.abstract_ops.LinalgStructuredOperation,
    tiled_loop_ivs: dict[int, SSAValue],
) -> None:
    """
    Give each `linalg.index` in a tiled body the index it read before tiling.

    A `linalg.index` reads the position of the iteration it runs in. The tiled op
    iterates over one tile, so it reads a position within that tile, and the
    offset the tile starts at is added back to it. A dimension that was not tiled
    is iterated over whole, so its index is already the one it was.
    """

    # A linalg.index can only be a direct child of the op whose iteration it
    # reads, so the block itself is enough to find every one of them. They are
    # taken before any is offset, since offsetting inserts into that same block,
    # and paired with the offset to give them, which a dimension that was not
    # tiled has none of.
    index_ops = tuple(
        (body_op, iv)
        for body_op in tiled_op.body.block.ops
        if isinstance(body_op, linalg.ops.IndexOp)
        if (iv := tiled_loop_ivs.get(body_op.dim.value.data)) is not None
    )

    for body_op, iv in index_ops:
        offset_op = affine.ApplyOp((body_op.result, iv), _INDEX_OFFSET_MAP)
        rewriter.insert(offset_op, InsertPoint.after(body_op))
        # Every reader of the index takes the offset one instead, other than the
        # op doing the offsetting, which is left reading the index itself.
        body_op.result.replace_uses_with_if(
            offset_op.result, lambda use: use.operation is not offset_op
        )


def tile_structured_op(
    rewriter: PatternRewriter,
    op: linalg.abstract_ops.LinalgStructuredOperation,
    tile_sizes: Sequence[SSAValue | int],
) -> bool:
    """
    Rewrite supported structured linalg ops into tiled form.
    """
    try:
        plan = TilingPlan.analyze(op, tile_sizes)
    except (ValueError, NotImplementedError) as e:
        raise PassFailedException(str(e)) from e

    if not plan.tiled_dims:
        return False

    # Outputs with value semantics are threaded through the loops, since each
    # tile produces a new value instead of writing through a view.
    has_tensor_outputs = bool(op.res)
    iter_args = tuple(op.outputs) if has_tensor_outputs else ()

    loops, tiled_loop_ivs, inner_ip = _build_tile_loops(
        rewriter,
        InsertPoint.before(op),
        plan.loop_ranges,
        plan.tile_sizes,
        plan.tiled_dims,
        _loop_range_sources(op, plan),
        iter_args,
    )

    num_inputs = len(op.inputs)
    effective_tile_sizes = _build_effective_tile_sizes(rewriter, inner_ip, plan, loops)

    slice_parameters = tuple(
        SliceParameters.compute(
            rewriter,
            inner_ip,
            indexing_map.data,
            operand_info,
            tiled_loop_ivs,
            effective_tile_sizes,
            plan.loop_ranges,
        )
        for operand_info, indexing_map in zip(
            plan.operand_infos, op.get_indexing_maps(), strict=True
        )
    )

    # Output tensors are sliced from the values the innermost loop carries, so
    # that each tile builds on the tiles the surrounding iterations wrote back.
    # Slicing the originals instead would discard their work.
    carried = loops[-1].body.block.args[1:]
    slice_sources = list(op.operands)
    slice_sources[num_inputs:] = carried if has_tensor_outputs else op.outputs

    tiled_operands = [
        _build_tiled_slice(
            rewriter, inner_ip, source, operand_info.source_type, parameters
        )
        for source, operand_info, parameters in zip(
            slice_sources, plan.operand_infos, slice_parameters, strict=True
        )
    ]

    tiled_outputs = tiled_operands[num_inputs:]
    # A tile is the same op over its slices, whatever op that is, so it is built
    # as one of those rather than as a generic. A named op says what it computes
    # by which op it is, which tiling it has no reason to take away from it.
    tiled_op = type(op).create(
        operands=tiled_operands,
        result_types=(
            tuple(value.type for value in tiled_outputs) if has_tensor_outputs else ()
        ),
        properties=dict(op.properties),
        regions=[op.body.clone()],
    )
    rewriter.insert(tiled_op, inner_ip)
    _offset_tiled_indices(rewriter, tiled_op, tiled_loop_ivs)

    # Memref outputs are written through their subview, so only tensor outputs
    # need their computed tile written back into the value being carried.
    yielded: Sequence[SSAValue] = (
        tuple(
            _build_tiled_insert(
                rewriter, inner_ip, tiled_result, destination, parameters
            )
            for tiled_result, destination, parameters in zip(
                tiled_op.res, carried, slice_parameters[num_inputs:], strict=True
            )
        )
        if has_tensor_outputs
        else ()
    )

    # The innermost loop yields the updated tensors, and each enclosing loop
    # yields the results of the loop nested inside it.
    for loop in reversed(loops):
        rewriter.insert(scf.YieldOp(*yielded), InsertPoint.at_end(loop.body.block))
        yielded = loop.results

    # The outermost loop carries out the fully updated tensors. An op tiling
    # memrefs carries nothing and has no results, so this replaces it with
    # nothing, which is the erase that case needs.
    rewriter.replace(op, [], loops[0].results)
    return True
