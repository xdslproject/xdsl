import re
from collections.abc import Sequence
from typing import Any

import pytest

from xdsl.builder import Builder
from xdsl.dialects import linalg, memref, tensor
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    DenseArrayBase,
    IndexType,
    MemRefType,
    ModuleOp,
    TensorType,
    f32,
    i64,
)
from xdsl.dialects.linalg.transforms.tiling import (
    OperandTileInfo,
    SliceParameters,
    TilingPlan,
    _build_tile_loops,  # pyright: ignore[reportPrivateUsage]
    _build_tiled_slice,  # pyright: ignore[reportPrivateUsage]
    tile_linalg_generic,
)
from xdsl.ir import Attribute
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa
from xdsl.utils.test_value import create_ssa_value


def test_operand_tile_info_analyze_identity_map():
    source_type = MemRefType(f32, [4, 5])
    indexing_map = AffineMap.from_callable(lambda i, j: (i, j))

    info = OperandTileInfo.analyze(indexing_map, source_type, (2, 0))

    assert info.source_type == source_type
    assert info.loop_dims == (0, 1)
    assert info.result_shape == (2, 5)


def test_operand_tile_info_analyze_transpose_map():
    source_type = MemRefType(f32, [5, 4])
    indexing_map = AffineMap.from_callable(lambda i, j: (j, i))

    info = OperandTileInfo.analyze(indexing_map, source_type, (2, 0))

    assert info.source_type == source_type
    assert info.loop_dims == (1, 0)
    assert info.result_shape == (5, 2)


def _generic_2d_copy_op(
    input_type: Attribute = MemRefType(f32, [4, 5]),
    output_type: Attribute = MemRefType(f32, [4, 5]),
    indexing_maps: Sequence[AffineMapAttr] | None = None,
    iterator_types: Sequence[linalg.attrs.IteratorTypeAttr] | None = None,
    result_types: Sequence[Attribute] = (),
    use_index: bool = False,
) -> linalg.ops.GenericOp:
    lhs = create_ssa_value(input_type)
    out = create_ssa_value(output_type)

    @Builder.implicit_region((f32, f32))
    def body(args: tuple[Any, ...]):
        if use_index:
            linalg.ops.IndexOp(0)
        linalg.ops.YieldOp(args[0])

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    if indexing_maps is None:
        indexing_maps = [
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ]

    if iterator_types is None:
        iterator_types = [
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
        ]

    return linalg.ops.GenericOp(
        [lhs],
        [out],
        body,
        indexing_maps,
        iterator_types,
        result_types,
    )


def test_tiling_plan_analyze_generic_op():
    op = _generic_2d_copy_op()

    plan = TilingPlan.analyze_generic_op(op, (2, 0))

    assert plan.loop_ranges == (4, 5)
    assert plan.tiled_dims == (0,)
    assert plan.tile_sizes == (2, 0)

    assert len(plan.operand_infos) == 2

    assert plan.operand_infos[0].loop_dims == (0, 1)
    assert plan.operand_infos[0].result_shape == (2, 5)

    assert plan.operand_infos[1].loop_dims == (0, 1)
    assert plan.operand_infos[1].result_shape == (2, 5)


def test_tiling_plan_analyze_generic_op_without_tiled_dims():
    op = _generic_2d_copy_op()

    plan = TilingPlan.analyze_generic_op(op, (0, 0))

    assert plan.loop_ranges == ()
    assert plan.tiled_dims == ()
    assert plan.operand_infos == ()
    assert plan.tile_sizes == (0, 0)


# Unsupported tiling analysis cases
def test_tiling_plan_rejects_negative_tile_size():
    op = _generic_2d_copy_op()

    with pytest.raises(ValueError, match="negative tile sizes"):
        TilingPlan.analyze_generic_op(op, (-1, 0))


def test_tiling_plan_rejects_tensor_results():
    op = _generic_2d_copy_op(result_types=(TensorType(f32, [4, 5]),))

    with pytest.raises(NotImplementedError, match="tensor results"):
        TilingPlan.analyze_generic_op(op, (2, 0))


def test_tiling_plan_rejects_linalg_index():
    op = _generic_2d_copy_op(use_index=True)

    with pytest.raises(ValueError, match=re.escape("using linalg.index")):
        TilingPlan.analyze_generic_op(op, (2, 0))


def test_tiling_plan_rejects_non_parallel_tiled_iterator():
    op = _generic_2d_copy_op(
        iterator_types=[
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.REDUCTION),
        ]
    )

    with pytest.raises(ValueError, match="non-parallel iterator dimensions"):
        TilingPlan.analyze_generic_op(op, (0, 2))


def test_tiling_plan_rejects_non_memref_operand():
    op = _generic_2d_copy_op(input_type=TensorType(f32, [4, 5]))

    with pytest.raises(NotImplementedError, match="non-memref operands"):
        TilingPlan.analyze_generic_op(op, (2, 0))


def test_tiling_plan_rejects_dynamic_operand_shape():
    op = _generic_2d_copy_op(input_type=MemRefType(f32, [DYNAMIC_INDEX, 5]))

    with pytest.raises(ValueError, match="dynamic operand shapes"):
        TilingPlan.analyze_generic_op(op, (2, 0))


def test_tiling_plan_rejects_non_projected_permutation_map():
    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    op = _generic_2d_copy_op(
        input_type=MemRefType(f32, [8]),
        indexing_maps=[
            AffineMapAttr(AffineMap(2, 0, (i + j,))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ],
    )

    with pytest.raises(ValueError, match="non-projected-permutation indexing maps"):
        TilingPlan.analyze_generic_op(op, (2, 0))


def test_tiling_plan_rejects_partial_tiles():
    op = _generic_2d_copy_op()

    with pytest.raises(ValueError, match="partial tiles"):
        TilingPlan.analyze_generic_op(op, (3, 0))


def test_slice_parameters_compute_tiled_and_untiled_dims():
    source_type = MemRefType(f32, [4, 5])
    iv = create_ssa_value(IndexType())
    indexing_map = AffineMap.from_callable(lambda i, j: (i, j))
    operand_info = OperandTileInfo.analyze(indexing_map, source_type, (2, 0))

    parameters = SliceParameters.compute(indexing_map, operand_info, {0: iv})

    # Dim 0's loop is tiled, so it starts at the induction variable and spans one
    # tile; dim 1's loop is not, so it starts at zero and spans the whole operand.
    assert parameters.offsets == (iv, 0)
    assert parameters.sizes == (2, 5)
    assert parameters.strides == (1, 1)


def test_slice_parameters_compute_without_tiled_dims():
    source_type = MemRefType(f32, [4, 5])
    indexing_map = AffineMap.from_callable(lambda i, j: (i, j))
    operand_info = OperandTileInfo.analyze(indexing_map, source_type, (0, 0))

    parameters = SliceParameters.compute(indexing_map, operand_info, {})

    # Nothing is tiled, so the slice covers the whole operand.
    assert parameters.offsets == (0, 0)
    assert parameters.sizes == (4, 5)


def test_slice_parameters_compute_follows_indexing_map():
    # The operand is indexed transposed, so loop dim 0 addresses the operand's
    # second dimension. The induction variable has to land there, not first.
    source_type = MemRefType(f32, [5, 4])
    iv = create_ssa_value(IndexType())
    indexing_map = AffineMap.from_callable(lambda i, j: (j, i))
    operand_info = OperandTileInfo.analyze(indexing_map, source_type, (2, 0))

    parameters = SliceParameters.compute(indexing_map, operand_info, {0: iv})

    assert parameters.offsets == (0, iv)
    assert parameters.sizes == (5, 2)


def _tiled_slice_for(source_type: MemRefType[Attribute] | TensorType[Attribute]):
    """Slice dim 0 of a 2d operand at a loop induction variable, tile size 2."""
    op = _generic_2d_copy_op()
    ModuleOp([op])
    rewriter = PatternRewriter(op)

    operand = create_ssa_value(source_type)
    iv = create_ssa_value(IndexType())
    indexing_map = AffineMap.from_callable(lambda i, j: (i, j))
    operand_info = OperandTileInfo.analyze(indexing_map, source_type, (2, 0))
    parameters = SliceParameters.compute(indexing_map, operand_info, {0: iv})

    result = _build_tiled_slice(
        rewriter, InsertPoint.before(op), operand, source_type, parameters
    )
    return result.owner


def test_build_tiled_slice_memref_emits_subview():
    slice_op = _tiled_slice_for(MemRefType(f32, [4, 5]))

    assert isinstance(slice_op, memref.SubviewOp)
    assert slice_op.static_offsets == DenseArrayBase.from_list(i64, [DYNAMIC_INDEX, 0])
    assert slice_op.static_sizes == DenseArrayBase.from_list(i64, [2, 5])
    assert len(slice_op.offsets) == 1
    assert isa(slice_op.result.type, MemRefType)
    assert slice_op.result.type.get_shape() == (2, 5)


def test_build_tiled_slice_tensor_emits_extract_slice():
    slice_op = _tiled_slice_for(TensorType(f32, [4, 5]))

    assert isinstance(slice_op, tensor.ExtractSliceOp)
    # Dim 0 is tiled, so its offset is the induction variable and its size the
    # tile size; dim 1 is untiled, so it takes the whole extent at offset 0.
    assert slice_op.static_offsets == DenseArrayBase.from_list(i64, [DYNAMIC_INDEX, 0])
    assert slice_op.static_sizes == DenseArrayBase.from_list(i64, [2, 5])
    assert slice_op.static_strides == DenseArrayBase.from_list(i64, [1, 1])
    assert len(slice_op.offsets) == 1
    assert slice_op.result.type == TensorType(f32, [2, 5])


def test_build_tile_loops_without_iter_args():
    op = _generic_2d_copy_op()
    ModuleOp([op])
    rewriter = PatternRewriter(op)

    loops, tiled_loop_ivs, _ = _build_tile_loops(
        rewriter, InsertPoint.before(op), (4, 5), (2, 0), (0,)
    )

    (loop,) = loops
    assert loop.iter_args == ()
    assert loop.results == ()
    assert loop.body.block.args == (tiled_loop_ivs[0],)


def test_build_tile_loops_threads_iter_args():
    op = _generic_2d_copy_op()
    ModuleOp([op])
    rewriter = PatternRewriter(op)

    init = create_ssa_value(TensorType(f32, [4, 5]))

    loops, tiled_loop_ivs, _ = _build_tile_loops(
        rewriter, InsertPoint.before(op), (4, 5), (2, 2), (0, 1), (init,)
    )

    outer, inner = loops

    # The outermost loop initialises the carried value from the original value,
    # and every nested loop from the enclosing loop's block argument. Initialising
    # a nested loop from the original would discard the surrounding iterations.
    assert outer.iter_args == (init,)
    assert inner.iter_args == (outer.body.block.args[1],)

    # Each loop carries the value back out as a result.
    assert outer.result_types == (init.type,)
    assert inner.result_types == (init.type,)

    # Body blocks gain one argument per carried value, after the induction variable.
    for loop, iv in ((outer, tiled_loop_ivs[0]), (inner, tiled_loop_ivs[1])):
        assert len(loop.body.block.args) == 2
        assert loop.body.block.args[0] is iv
        assert loop.body.block.args[1].type == init.type


def test_tile_linalg_generic_returns_false_without_tiled_dims():
    op = _generic_2d_copy_op()
    ModuleOp([op])
    rewriter = PatternRewriter(op)

    assert not tile_linalg_generic(rewriter, op, (0, 0))
