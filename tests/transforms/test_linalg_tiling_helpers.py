from collections.abc import Sequence
from typing import Any

import pytest

from xdsl.builder import Builder
from xdsl.dialects import linalg
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    MemRefType,
    TensorType,
    f32,
)
from xdsl.dialects.linalg.transforms.tiling import OperandTileInfo, TilingPlan
from xdsl.ir import Attribute
from xdsl.ir.affine import AffineExpr, AffineMap
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

    with pytest.raises(ValueError, match="using linalg.index"):
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
