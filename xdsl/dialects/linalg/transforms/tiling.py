from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import assert_never

from xdsl.dialects import arith, linalg, memref, scf, tensor
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    MemRefType,
    TensorType,
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
    - `result_shape` the shape that tiled slice should have.
    """

    source_type: MemRefType[Attribute] | TensorType[Attribute]
    loop_dims: tuple[int, ...]
    result_shape: tuple[int, ...]

    @staticmethod
    def analyze(
        indexing_map: AffineMap,
        source_type: MemRefType[Attribute] | TensorType[Attribute],
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
            assert isa(source_type, MemRefType | TensorType)
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


def _build_tile_loops(
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
    loop_ranges: Sequence[int],
    tile_sizes: Sequence[int],
    tiled_dims: Sequence[int],
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
        - `current_insertion_point`: the place to insert `tiled subview` and the `tiled generic`
    """

    index = IndexType()
    zero = arith.ConstantOp(IntegerAttr(0, index))
    ub_ops = {
        dim: arith.ConstantOp(IntegerAttr(loop_ranges[dim], index))
        for dim in tiled_dims
    }
    tile_ops = {
        dim: arith.ConstantOp(IntegerAttr(tile_sizes[dim], index)) for dim in tiled_dims
    }
    rewriter.insert(
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
    carried_types = tuple(value.type for value in iter_args)
    carried: Sequence[SSAValue] = iter_args
    for dim in tiled_dims:
        loop = scf.ForOp(
            zero.result,
            ub_ops[dim].result,
            tile_ops[dim].result,
            carried,
            Region(Block(arg_types=(index, *carried_types))),
        )
        rewriter.insert(loop, current_insertion_point)
        loops.append(loop)
        tiled_loop_ivs[dim] = loop.body.block.args[0]
        carried = loop.body.block.args[1:]
        current_insertion_point = InsertPoint.at_start(loop.body.block)

    return loops, tiled_loop_ivs, current_insertion_point


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
        indexing_map: AffineMap,
        operand_info: OperandTileInfo,
        tiled_loop_ivs: dict[int, SSAValue],
    ) -> "SliceParameters":
        """
        Compute where the current tile sits within one operand.

        Each result of the indexing map addresses one dimension of the operand.
        A dimension whose loop is tiled starts at that loop's induction variable
        and spans one tile, and a dimension whose loop is not tiled starts at
        zero and spans the whole operand.
        """

        source_shape = operand_info.source_type.get_shape()

        offsets: list[SSAValue | int] = []
        sizes: list[SSAValue | int] = []
        for result_index, expr in enumerate(indexing_map.results):
            assert isinstance(expr, AffineDimExpr)
            loop_dim = operand_info.loop_dims[result_index]
            if loop_dim in tiled_loop_ivs:
                offsets.append(tiled_loop_ivs[loop_dim])
                sizes.append(operand_info.result_shape[result_index])
            else:
                offsets.append(0)
                sizes.append(source_shape[result_index])

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


def tile_linalg_generic(
    rewriter: PatternRewriter,
    op: linalg.ops.GenericOp,
    tile_sizes: tuple[int, ...],
) -> bool:
    """
    Rewrite supported `linalg.generic` ops into tiled formed.
    """
    try:
        plan = TilingPlan.analyze_generic_op(op, tile_sizes)
    except (ValueError, NotImplementedError) as e:
        raise PassFailedException(str(e)) from e

    if not plan.tiled_dims:
        return False

    loops, tiled_loop_ivs, inner_ip = _build_tile_loops(
        rewriter,
        InsertPoint.before(op),
        plan.loop_ranges,
        plan.tile_sizes,
        plan.tiled_dims,
    )
    tiled_operands: list[SSAValue] = []

    for operand, operand_info, indexing_map in zip(
        op.operands, plan.operand_infos, op.get_indexing_maps(), strict=True
    ):
        parameters = SliceParameters.compute(
            indexing_map.data, operand_info, tiled_loop_ivs
        )
        tiled_operands.append(
            _build_tiled_slice(
                rewriter,
                inner_ip,
                operand,
                operand_info.source_type,
                parameters,
            )
        )

    num_inputs = len(op.inputs)
    tiled_generic = linalg.ops.GenericOp(
        tiled_operands[:num_inputs],
        tiled_operands[num_inputs:],
        op.body.clone(),
        op.get_indexing_maps(),
        op.get_iterator_types(),
    )
    rewriter.insert(tiled_generic, inner_ip)

    for loop in reversed(loops):
        rewriter.insert(scf.YieldOp(), InsertPoint.at_end(loop.body.block))

    rewriter.erase(op)
    return True
