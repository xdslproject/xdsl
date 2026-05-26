from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.dialects import linalg
from xdsl.dialects.builtin import MemRefType
from xdsl.ir import Attribute
from xdsl.ir.affine import AffineDimExpr, AffineMap
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

    @staticmethod
    def analyze(
        indexing_map: AffineMap,
        source_type: MemRefType[Attribute],
        tile_sizes: Sequence[int],
    ) -> "OperandTileInfo":
        """
        Analyze how one operand should be sliced for each tile.
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

    @staticmethod
    def analyze_generic_op(
        op: linalg.ops.GenericOp,
        tile_sizes: tuple[int, ...],
    ) -> "TilingPlan":
        """
        Analyze one supported `linalg.generic` and return a `TilingPlan`.
        """

        num_loops = op.get_num_loops()
        normalized_tile_sizes = tile_sizes[:num_loops] + (0,) * (
            num_loops - len(tile_sizes)
        )

        tiled_dims = tuple(
            dim for dim, tile_size in enumerate(normalized_tile_sizes) if tile_size != 0
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
        for operand, indexing_map in zip(
            op.operands, op.get_indexing_maps(), strict=True
        ):
            source_type = operand.type
            assert isa(source_type, MemRefType)
            operand_infos_list.append(
                OperandTileInfo.analyze(
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


def _verify_generic_is_tileable(
    op: linalg.ops.GenericOp,
    tile_sizes: Sequence[int],
    tiled_dims: Sequence[int],
) -> tuple[int, ...]:
    """
    Check whether a `linalg.generic` is safe to tile.
    """

    if any(tile_sizes[dim] < 0 for dim in tiled_dims):
        raise ValueError("negative tile sizes are not supported")

    if op.res:
        raise NotImplementedError(
            "tiling linalg.generic with tensor results is not supported yet"
        )

    if any(isa(body_op, linalg.ops.IndexOp) for body_op in op.body.walk()):
        raise ValueError(
            "tiling linalg.generic using linalg.index is not supported yet"
        )

    iterator_types = tuple(iterator.data for iterator in op.get_iterator_types())
    if any(
        iterator_types[dim] != linalg.attrs.IteratorType.PARALLEL for dim in tiled_dims
    ):
        raise ValueError(
            "tiling of non-parallel iterator dimensions is not supported yet"
        )

    indexing_maps = tuple(attr.data for attr in op.get_indexing_maps())
    for operand, indexing_map in zip(op.operands, indexing_maps, strict=True):
        raw_operand_type = operand.type

        if not isa(raw_operand_type, MemRefType):
            raise NotImplementedError(
                "tiling linalg.generic with non-memref operands is not supported yet"
            )
        operand_type = raw_operand_type

        if any(dim < 0 for dim in operand_type.get_shape()):
            raise ValueError(
                "tiling linalg.generic with dynamic operand shapes is not supported yet"
            )

        if not indexing_map.is_projected_permutation():
            raise ValueError(
                "tiling linalg.generic with non-projected-permutation indexing maps is not supported yet"
            )

    loop_ranges = op.get_static_loop_ranges()
    if any(loop_ranges[dim] % tile_sizes[dim] for dim in tiled_dims):
        raise ValueError("partial tiles are not supported yet")

    return loop_ranges
