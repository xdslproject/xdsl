from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.dialects import arith, linalg, memref, scf
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    MemRefType,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class OperandTileInfo:
    """
    This records how one operand should be sliced when we enter a tile.
    - `source_type` keeps the original type.
    - `loop_dims` the loop dimension that comes from each indexing-map.
    - `result_shape` the shape that tiled subview should have.
    """

    source_type: MemRefType[Attribute]
    loop_dims: tuple[int, ...]
    result_shape: tuple[int, ...]


@dataclass(frozen=True)
class TilingPlan:
    """
    This stores the information needed to turn one op into tiled loop and tiled subview.
    - `loop_ranges` are original static loop ranges.
    - `tiled_dims` the dimensions that really get tiled.
    - `operand_infos` stores one `OperandTileInfo` per operand.
    - `tile_sizes` are the normalized tile sizes, padded to match the op loop count.
    """

    loop_ranges: tuple[int, ...]
    tiled_dims: tuple[int, ...]
    operand_infos: tuple[OperandTileInfo, ...]
    tile_sizes: tuple[int, ...]


def _normalize_tile_sizes(
    tile_sizes: tuple[int, ...], num_loops: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Pad tile sizes to match the loop count.
    A tile size of `0` means "do not tile this dimension".
    """

    normalized = tile_sizes[:num_loops] + (0,) * (num_loops - len(tile_sizes))
    tiled_dims = tuple(dim for dim, size in enumerate(normalized) if size != 0)
    return normalized, tiled_dims


def _verify_generic_is_tileable(
    op: linalg.ops.GenericOp,
    tile_sizes: Sequence[int],
    tiled_dims: Sequence[int],
) -> tuple[int, ...]:
    """
    Check whether a `linalg.generic` is safe to tile.
    """

    if any(tile_sizes[dim] < 0 for dim in tiled_dims):
        raise PassFailedException("negative tile sizes are not supported")

    if op.res:
        raise PassFailedException(
            "tiling linalg.generic with tensor results is not supported yet"
        )

    if any(isa(body_op, linalg.ops.IndexOp) for body_op in op.body.walk()):
        raise PassFailedException(
            "tiling linalg.generic using linalg.index is not supported yet"
        )

    iterator_types = tuple(iterator.data for iterator in op.get_iterator_types())
    if any(
        iterator_types[dim] != linalg.attrs.IteratorType.PARALLEL for dim in tiled_dims
    ):
        raise PassFailedException(
            "tiling of non-parallel iterator dimensions is not supported yet"
        )

    indexing_maps = tuple(attr.data for attr in op.get_indexing_maps())
    for operand, indexing_map in zip(op.operands, indexing_maps, strict=True):
        raw_operand_type = operand.type

        if not isa(raw_operand_type, MemRefType):
            raise PassFailedException(
                "tiling linalg.generic with non-memref operands is not supported yet"
            )
        operand_type = raw_operand_type

        if any(dim < 0 for dim in operand_type.get_shape()):
            raise PassFailedException(
                "tiling linalg.generic with dynamic operand shapes is not supported yet"
            )

        if not indexing_map.is_projected_permutation():
            raise PassFailedException(
                "tiling linalg.generic with non-projected-permutation indexing maps is not supported yet"
            )

    loop_ranges = op.get_static_loop_ranges()
    if any(loop_ranges[dim] % tile_sizes[dim] for dim in tiled_dims):
        raise PassFailedException("partial tiles are not supported yet")

    return loop_ranges


def _build_tile_loops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    loop_ranges: Sequence[int],
    tile_sizes: Sequence[int],
    tiled_dims: Sequence[int],
) -> tuple[list[scf.ForOp], dict[int, SSAValue], InsertPoint]:
    """
    Build the outer tiled loops.

    Return:
        - `loops`: the outer `scf.for` ops
        - `tiled_loop_ivs`: a map from loop dimensions to induction variables
        - `current_insertion_point`: the place to insert `tiled subview` and the `tiled generic`
    """

    zero = arith.ConstantOp(IntegerAttr.from_index_int_value(0))
    ub_ops = {
        dim: arith.ConstantOp(IntegerAttr.from_index_int_value(loop_ranges[dim]))
        for dim in tiled_dims
    }
    tile_ops = {
        dim: arith.ConstantOp(IntegerAttr.from_index_int_value(tile_sizes[dim]))
        for dim in tiled_dims
    }
    rewriter.insert_op(
        [
            zero,
            *(ub_ops[dim] for dim in tiled_dims),
            *(tile_ops[dim] for dim in tiled_dims),
        ],
        insertion_point,
    )

    current_insertion_point = insertion_point
    loops: list[scf.ForOp] = []
    tiled_loop_ivs: dict[int, SSAValue] = {}
    for dim in tiled_dims:
        loop = scf.ForOp(
            zero.result,
            ub_ops[dim].result,
            tile_ops[dim].result,
            (),
            Region(Block(arg_types=(IndexType(),))),
        )
        rewriter.insert_op(loop, current_insertion_point)
        loops.append(loop)
        tiled_loop_ivs[dim] = loop.body.block.args[0]
        current_insertion_point = InsertPoint.at_start(loop.body.block)

    return loops, tiled_loop_ivs, current_insertion_point


def _build_tiled_subview_type(
    source_type: MemRefType, result_shape: Sequence[int]
) -> MemRefType:
    """
    Build `the type` for one tiled subview.
    """

    layout = source_type.layout
    if not isinstance(layout, (NoneAttr, StridedLayoutAttr)):
        raise PassFailedException(
            f"tiling memrefs with layout {layout} is not supported yet"
        )

    strides = source_type.get_strides()
    assert strides is not None
    if any(stride is None for stride in strides):
        raise PassFailedException(
            "tiling memrefs with dynamic strides is not supported yet"
        )

    layout = StridedLayoutAttr(tuple(strides), None)

    return MemRefType(
        source_type.element_type,
        result_shape,
        layout,
        source_type.memory_space,
    )


def _build_tiled_subview(
    operand: SSAValue,
    indexing_map: AffineMap,
    operand_info: OperandTileInfo,
    tiled_loop_ivs: dict[int, SSAValue],
) -> memref.SubviewOp:
    """
    Build `the subview` for one operand at the current tile position.
    """

    source_shape = operand_info.source_type.get_shape()

    offsets: list[SSAValue | int] = []
    sizes: list[int] = []
    for result_index, expr in enumerate(indexing_map.results):
        assert isinstance(expr, AffineDimExpr)
        loop_dim = operand_info.loop_dims[result_index]
        if loop_dim in tiled_loop_ivs:
            offsets.append(tiled_loop_ivs[loop_dim])
            sizes.append(operand_info.result_shape[result_index])
        else:
            offsets.append(0)
            sizes.append(source_shape[result_index])

    return memref.SubviewOp.get(
        operand,
        offsets,
        sizes,
        (1,) * len(source_shape),
        _build_tiled_subview_type(operand_info.source_type, sizes),
    )


def _analyze_operand_tile_info(
    indexing_map: AffineMap,
    source_type: MemRefType[Attribute],
    tile_sizes: Sequence[int],
) -> OperandTileInfo:
    """
    Analyze how one operand should be sliced for each tile, and returned an `OperandTileInfo`.
    """

    source_shape = source_type.get_shape()
    loop_dims = tuple(
        cast(AffineDimExpr, expr).position for expr in indexing_map.results
    )
    result_shape = tuple(
        tile_sizes[loop_dim]
        if tile_sizes[loop_dim] != 0
        else source_shape[result_index]
        for result_index, loop_dim in enumerate(loop_dims)
    )
    return OperandTileInfo(source_type, loop_dims, result_shape)


def _analyze_generic_op(
    op: linalg.ops.GenericOp,
    tile_sizes: tuple[int, ...],
) -> TilingPlan:
    """
    Analyze one supported `linalg.generic` and returned a `TilingPlan`.
    """

    normalized_tile_sizes, tiled_dims = _normalize_tile_sizes(
        tile_sizes, op.get_num_loops()
    )

    if not tiled_dims:
        return TilingPlan(
            loop_ranges=(),
            tiled_dims=(),
            operand_infos=(),
            tile_sizes=normalized_tile_sizes,
        )

    loop_ranges = _verify_generic_is_tileable(
        op,
        normalized_tile_sizes,
        tiled_dims,
    )

    operand_infos_list: list[OperandTileInfo] = []
    for operand, indexing_map in zip(op.operands, op.get_indexing_maps(), strict=True):
        source_type = operand.type
        assert isa(source_type, MemRefType)
        operand_infos_list.append(
            _analyze_operand_tile_info(
                indexing_map.data,
                source_type,
                normalized_tile_sizes,
            )
        )
    operand_infos = tuple(operand_infos_list)

    return TilingPlan(
        loop_ranges=loop_ranges,
        tiled_dims=tiled_dims,
        operand_infos=operand_infos,
        tile_sizes=normalized_tile_sizes,
    )


def tile_linalg_generic(
    rewriter: PatternRewriter,
    op: linalg.ops.GenericOp,
    tile_sizes: tuple[int, ...],
) -> bool:
    """
    Rewrite supported `linalg.generic` ops into tiled formed.
    """
    plan = _analyze_generic_op(op, tile_sizes)
    if not plan.tiled_dims:
        return False

    loops, tiled_loop_ivs, inner_ip = _build_tile_loops(
        rewriter,
        InsertPoint.before(op),
        plan.loop_ranges,
        plan.tile_sizes,
        plan.tiled_dims,
    )
    tiled_subviews: list[memref.SubviewOp] = []
    tiled_operands: list[SSAValue] = []

    for operand, operand_info, indexing_map in zip(
        op.operands, plan.operand_infos, op.get_indexing_maps(), strict=True
    ):
        subview = _build_tiled_subview(
            operand, indexing_map.data, operand_info, tiled_loop_ivs
        )
        tiled_subviews.append(subview)
        tiled_operands.append(subview.result)

    rewriter.insert_op(tiled_subviews, inner_ip)

    num_inputs = len(op.inputs)
    tiled_generic = linalg.GenericOp(
        tiled_operands[:num_inputs],
        tiled_operands[num_inputs:],
        op.body.clone(),
        op.get_indexing_maps(),
        op.get_iterator_types(),
    )
    rewriter.insert_op(tiled_generic, InsertPoint.after(tiled_subviews[-1]))

    for loop in reversed(loops):
        rewriter.insert_op(scf.YieldOp(), InsertPoint.at_end(loop.body.block))

    rewriter.erase_op(op)
    return True
