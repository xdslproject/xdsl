import pytest

from xdsl.dialects.builtin import f32, f64, i32, i64
from xdsl.dialects.wasmssa import (
    AddOp,
    AndOp,
    BinaryComparisonOperation,
    BinaryNumericalOperation,
    CopySignOp,
    DivOp,
    DivSIOp,
    DivUIOp,
    EqOp,
    EqzOp,
    GeOp,
    GeSIOp,
    GeUIOp,
    GtOp,
    GtSIOp,
    GtUIOp,
    LeOp,
    LeSIOp,
    LeUIOp,
    LtOp,
    LtSIOp,
    LtUIOp,
    MaxOp,
    MinOp,
    MulOp,
    NeOp,
    OrOp,
    RemSIOp,
    RemUIOp,
    SubOp,
    XOrOp,
)
from xdsl.ir import Attribute
from xdsl.traits import Commutative, NoMemoryEffect, OpTrait, Pure
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize(
    "op_type, operand_type, expected_traits",
    [
        (AddOp, i32, [Pure(), Commutative()]),
        (AndOp, i32, [Pure(), Commutative()]),
        (DivOp, f32, [Pure()]),
        (DivUIOp, i32, [NoMemoryEffect()]),
        (DivSIOp, i32, [NoMemoryEffect()]),
        (MulOp, i32, [Pure(), Commutative()]),
        (OrOp, i32, [Pure(), Commutative()]),
        (SubOp, i32, [Pure()]),
        (RemUIOp, i32, [NoMemoryEffect()]),
        (RemSIOp, i32, [NoMemoryEffect()]),
        (XOrOp, i32, [Pure(), Commutative()]),
        (MinOp, f32, [Pure(), Commutative()]),
        (MaxOp, f32, [Pure(), Commutative()]),
        (CopySignOp, f32, [Pure()]),
    ],
)
def test_binary_numerical_operation_traits(
    op_type: type[BinaryNumericalOperation],
    operand_type: Attribute,
    expected_traits: list[OpTrait],
):
    lhs = create_ssa_value(operand_type)
    rhs = create_ssa_value(operand_type)
    op = op_type(lhs, rhs)

    assert op.traits.traits == frozenset(expected_traits)


@pytest.mark.parametrize("op_type", [EqOp, NeOp])
@pytest.mark.parametrize("operand_type", [i32, i64, f32, f64])
def test_numeric_comparison_construction(
    op_type: type[BinaryComparisonOperation],
    operand_type: Attribute,
):
    lhs = create_ssa_value(operand_type)
    rhs = create_ssa_value(operand_type)

    op = op_type(lhs, rhs)

    op.verify()
    assert op.result.type == i32


@pytest.mark.parametrize(
    "op_type",
    [LtSIOp, LtUIOp, LeSIOp, LeUIOp, GtSIOp, GtUIOp, GeSIOp, GeUIOp],
)
@pytest.mark.parametrize("operand_type", [i32, i64])
def test_integer_comparison_construction(
    op_type: type[BinaryComparisonOperation],
    operand_type: Attribute,
):
    lhs = create_ssa_value(operand_type)
    rhs = create_ssa_value(operand_type)

    op = op_type(lhs, rhs)

    op.verify()
    assert op.result.type == i32


@pytest.mark.parametrize("op_type", [LtOp, LeOp, GtOp, GeOp])
@pytest.mark.parametrize("operand_type", [f32, f64])
def test_float_comparison_construction(
    op_type: type[BinaryComparisonOperation],
    operand_type: Attribute,
):
    lhs = create_ssa_value(operand_type)
    rhs = create_ssa_value(operand_type)

    op = op_type(lhs, rhs)

    op.verify()
    assert op.result.type == i32


@pytest.mark.parametrize("operand_type", [i32, i64])
def test_eqz_construction(operand_type: Attribute):
    input = create_ssa_value(operand_type)

    op = EqzOp(input)

    op.verify()
    assert op.result.type == i32


@pytest.mark.parametrize(
    "op_type, expected_traits",
    [
        (EqOp, [Pure(), Commutative()]),
        (NeOp, [Pure(), Commutative()]),
        (LtSIOp, [Pure()]),
        (LtUIOp, [Pure()]),
        (LeSIOp, [Pure()]),
        (LeUIOp, [Pure()]),
        (GtSIOp, [Pure()]),
        (GtUIOp, [Pure()]),
        (GeSIOp, [Pure()]),
        (GeUIOp, [Pure()]),
        (LtOp, [Pure()]),
        (LeOp, [Pure()]),
        (GtOp, [Pure()]),
        (GeOp, [Pure()]),
    ],
)
def test_binary_comparison_traits(
    op_type: type[BinaryComparisonOperation],
    expected_traits: list[OpTrait],
):
    assert op_type.traits.traits == frozenset(expected_traits)


def test_eqz_traits():
    assert EqzOp.traits.traits == frozenset([Pure()])
