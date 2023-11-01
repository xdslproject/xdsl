import pytest

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineExpr
from xdsl.transforms.lower_affine import affine_expr_ops
from xdsl.utils.test_value import TestSSAValue


@pytest.mark.parametrize(
    "expr,dims,symbols,ops,val",
    [
        (
            AffineExpr.constant(42),
            [],
            [],
            [arith.Constant(IntegerAttr.from_index_int_value(42))],
            None,
        ),
        (
            AffineExpr.dimension(0),
            [value0 := TestSSAValue(IndexType())],
            [],
            [],
            value0,
        ),
        (
            AffineExpr.symbol(0),
            [],
            [value1 := TestSSAValue(IndexType())],
            [],
            value1,
        ),
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Add, AffineExpr.dimension(0), AffineExpr.dimension(1)
            ),
            [value2 := TestSSAValue(IndexType()), value3 := TestSSAValue(IndexType())],
            [],
            [arith.Addi(value2, value3)],
            None,
        ),
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Mul, AffineExpr.dimension(0), AffineExpr.dimension(1)
            ),
            [value2 := TestSSAValue(IndexType()), value3 := TestSSAValue(IndexType())],
            [],
            [arith.Muli(value2, value3)],
            None,
        ),
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Mod, AffineExpr.dimension(0), AffineExpr.dimension(1)
            ),
            [value2 := TestSSAValue(IndexType()), value3 := TestSSAValue(IndexType())],
            [],
            [arith.RemSI(value2, value3)],
            None,
        ),
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.FloorDiv,
                AffineExpr.dimension(0),
                AffineExpr.dimension(1),
            ),
            [value2 := TestSSAValue(IndexType()), value3 := TestSSAValue(IndexType())],
            [],
            [arith.FloorDivSI(value2, value3)],
            None,
        ),
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.CeilDiv,
                AffineExpr.dimension(0),
                AffineExpr.dimension(1),
            ),
            [value2 := TestSSAValue(IndexType()), value3 := TestSSAValue(IndexType())],
            [],
            [arith.CeilDivSI(value2, value3)],
            None,
        ),
    ],
)
def test_affine_map_constant_ops(
    expr: AffineExpr,
    dims: list[SSAValue],
    symbols: list[SSAValue],
    ops: list[Operation],
    val: SSAValue | None,
):
    expr_ops, expr_val = affine_expr_ops(expr, dims, symbols)
    assert len(ops) == len(expr_ops)
    for op, expr_op in zip(ops, expr_ops):
        assert op.is_structurally_equivalent(expr_op)
    if val is None:
        assert expr_val is SSAValue.get(expr_ops[-1])
    else:
        assert val is expr_val
