from typing import Any

from xdsl.builder import Builder
from xdsl.dialects import linalg
from xdsl.dialects.builtin import AffineMapAttr, MemRefType, f32
from xdsl.dialects.linalg.transforms.tiling import OperandTileInfo, TilingPlan
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


def _generic_2d_copy_op() -> linalg.ops.GenericOp:
    lhs = create_ssa_value(MemRefType(f32, [4, 5]))
    out = create_ssa_value(MemRefType(f32, [4, 5]))

    @Builder.implicit_region((f32, f32))
    def body(args: tuple[Any, ...]):
        linalg.ops.YieldOp(args[0])

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    return linalg.ops.GenericOp(
        [lhs],
        [out],
        body,
        [
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ],
        [
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
        ],
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
