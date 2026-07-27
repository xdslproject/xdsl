import pytest

from xdsl.dialects.builtin import f32, i32
from xdsl.dialects.wasmssa import (
    AddOp,
    AndOp,
    BinaryNumericalOperation,
    CopySignOp,
    DivOp,
    DivSIOp,
    DivUIOp,
    MaxOp,
    MinOp,
    MulOp,
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
