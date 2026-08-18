from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import assert_never

from xdsl.dialects import arith, linalg, memref, scf, tensor
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IndexType,
    IntegerAttr,
    MemRefType,
    TensorType,
)
from xdsl.dialects.utils import split_dynamic_index_list
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
    """

    source_type: MemRefType[Attribute] | TensorType[Attribute]
    loop_dims: tuple[int, ...]

    @staticmethod
    def analyze(
        indexing_map: AffineMap,
        source_type: MemRefType[Attribute] | TensorType[Attribute],
    ) -> "OperandTileInfo":
        """
        Analyze how one operand should be sliced for each tile.
        """

        loop_dims = tuple(
            cast(AffineDimExpr, expr).position for expr in indexing_map.results
        )
        return OperandTileInfo(source_type, loop_dims)


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
                OperandTileInfo.analyze(indexing_map.data, source_type)
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

    # Operands that mix the two are rejected outright rather than tiled, since
    # each kind is written back a different way. MLIR does not consider such an
    # op valid either, requiring pure tensor or pure buffer semantics.
    operand_types = tuple(operand.type for operand in op.operands)
    if any(isa(operand_type, MemRefType) for operand_type in operand_types) and any(
        isa(operand_type, TensorType) for operand_type in operand_types
    ):
        raise ValueError(
            "tiling linalg.generic with a mix of memref and tensor operands is "
            "not supported"
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

        if not isa(raw_operand_type, MemRefType) and not isa(
            raw_operand_type, TensorType
        ):
            raise NotImplementedError(
                "tiling linalg.generic with operands that are neither memrefs nor "
                "tensors is not supported yet"
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
        tile_sizes: dict[int, SSAValue | int],
    ) -> "SliceParameters":
        """
        Compute where the current tile sits within one operand.

        Each result of the indexing map addresses one dimension of the operand.
        A dimension whose loop is tiled starts at that loop's induction variable
        and spans that loop's tile, and a dimension whose loop is not tiled
        starts at zero and spans the whole operand.
        """

        source_shape = operand_info.source_type.get_shape()

        offsets: list[SSAValue | int] = []
        sizes: list[SSAValue | int] = []
        for result_index, expr in enumerate(indexing_map.results):
            assert isinstance(expr, AffineDimExpr)
            loop_dim = operand_info.loop_dims[result_index]
            if loop_dim in tiled_loop_ivs:
                offsets.append(tiled_loop_ivs[loop_dim])
                sizes.append(tile_sizes[loop_dim])
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
        iter_args,
    )

    num_inputs = len(op.inputs)
    tile_sizes_by_dim: dict[int, SSAValue | int] = {
        dim: plan.tile_sizes[dim] for dim in plan.tiled_dims
    }

    slice_parameters = tuple(
        SliceParameters.compute(
            indexing_map.data, operand_info, tiled_loop_ivs, tile_sizes_by_dim
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
    tiled_generic = linalg.ops.GenericOp(
        tiled_operands[:num_inputs],
        tiled_outputs,
        op.body.clone(),
        op.get_indexing_maps(),
        op.get_iterator_types(),
        tuple(value.type for value in tiled_outputs) if has_tensor_outputs else (),
    )
    rewriter.insert(tiled_generic, inner_ip)

    # Memref outputs are written through their subview, so only tensor outputs
    # need their computed tile written back into the value being carried.
    yielded: Sequence[SSAValue] = (
        tuple(
            _build_tiled_insert(
                rewriter, inner_ip, tiled_result, destination, parameters
            )
            for tiled_result, destination, parameters in zip(
                tiled_generic.res, carried, slice_parameters[num_inputs:], strict=True
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
