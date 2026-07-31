import pytest

from xdsl.dialects.builtin import f32, i32
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
    RotlOp,
    RotrOp,
    ShiftRotateOperation,
    ShLOp,
    ShRSOp,
    ShRUOp,
    SubOp,
    XOrOp,
)
from xdsl.ir import Attribute
from xdsl.irdl import IRDLOperation
from xdsl.traits import Commutative, NoMemoryEffect, OpTrait, Pure
from xdsl.utils.test_value import create_ssa_value


def _assert_traits(
    op: IRDLOperation,
    required_traits: list[OpTrait],
    forbidden_traits: list[OpTrait],
):
    for trait in required_traits:
        assert op.has_trait(trait)
    for trait in forbidden_traits:
        assert not op.has_trait(trait)


@pytest.mark.parametrize(
    "op_type, operand_type, required_traits, forbidden_traits",
    [
        (AddOp, i32, [Pure(), Commutative()], []),
        (AndOp, i32, [Pure(), Commutative()], []),
        (DivOp, f32, [Pure()], [Commutative()]),
        (DivUIOp, i32, [NoMemoryEffect()], [Pure(), Commutative()]),
        (DivSIOp, i32, [NoMemoryEffect()], [Pure(), Commutative()]),
        (MulOp, i32, [Pure(), Commutative()], []),
        (OrOp, i32, [Pure(), Commutative()], []),
        (SubOp, i32, [Pure()], [Commutative()]),
        (RemUIOp, i32, [NoMemoryEffect()], [Pure(), Commutative()]),
        (RemSIOp, i32, [NoMemoryEffect()], [Pure(), Commutative()]),
        (XOrOp, i32, [Pure(), Commutative()], []),
        (MinOp, f32, [Pure(), Commutative()], []),
        (MaxOp, f32, [Pure(), Commutative()], []),
        (CopySignOp, f32, [Pure()], [Commutative()]),
        (ShLOp, i32, [Pure()], [Commutative()]),
        (ShRSOp, i32, [Pure()], [Commutative()]),
        (ShRUOp, i32, [Pure()], [Commutative()]),
        (RotlOp, i32, [Pure()], [Commutative()]),
        (RotrOp, i32, [Pure()], [Commutative()]),
        (EqOp, i32, [Pure(), Commutative()], []),
        (NeOp, i32, [Pure(), Commutative()], []),
        (LtSIOp, i32, [Pure()], [Commutative()]),
        (LtUIOp, i32, [Pure()], [Commutative()]),
        (LeSIOp, i32, [Pure()], [Commutative()]),
        (LeUIOp, i32, [Pure()], [Commutative()]),
        (GtSIOp, i32, [Pure()], [Commutative()]),
        (GtUIOp, i32, [Pure()], [Commutative()]),
        (GeSIOp, i32, [Pure()], [Commutative()]),
        (GeUIOp, i32, [Pure()], [Commutative()]),
        (LtOp, f32, [Pure()], [Commutative()]),
        (LeOp, f32, [Pure()], [Commutative()]),
        (GtOp, f32, [Pure()], [Commutative()]),
        (GeOp, f32, [Pure()], [Commutative()]),
    ],
)
def test_binary_traits(
    op_type: (
        type[BinaryNumericalOperation]
        | type[BinaryComparisonOperation]
        | type[ShiftRotateOperation]
    ),
    operand_type: Attribute,
    required_traits: list[OpTrait],
    forbidden_traits: list[OpTrait],
):
    lhs = create_ssa_value(operand_type)
    rhs = create_ssa_value(operand_type)

    _assert_traits(op_type(lhs, rhs), required_traits, forbidden_traits)


def test_eqz_traits():
    input = create_ssa_value(i32)

    _assert_traits(EqzOp(input), [Pure()], [Commutative()])
