import pytest

from xdsl.dialects.builtin import IntegerAttr, f32, f64, i32, i64
from xdsl.dialects.wasmssa import (
    AbsOp,
    AddOp,
    AndOp,
    BinaryComparisonOperation,
    BinaryNumericalOperation,
    CeilOp,
    ClzOp,
    ConversionOperation,
    ConvertSOp,
    ConvertUOp,
    CopySignOp,
    CtzOp,
    DemoteOp,
    DivOp,
    DivSIOp,
    DivUIOp,
    EqOp,
    EqzOp,
    ExtendLowBitsSOp,
    ExtendSI32Op,
    ExtendUI32Op,
    FloorOp,
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
    NegOp,
    NeOp,
    NumericType,
    OrOp,
    PopCntOp,
    PromoteOp,
    ReinterpretOp,
    RemSIOp,
    RemUIOp,
    RotlOp,
    RotrOp,
    ShiftRotateOperation,
    ShLOp,
    ShRSOp,
    ShRUOp,
    SqrtOp,
    SubOp,
    TruncOp,
    UnaryNumericalOperation,
    WrapOp,
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


@pytest.mark.parametrize(
    "op_type, operand_type",
    [
        (AbsOp, f32),
        (CeilOp, f32),
        (FloorOp, f32),
        (NegOp, f32),
        (SqrtOp, f32),
        (TruncOp, f32),
        (ClzOp, i32),
        (CtzOp, i32),
        (EqzOp, i32),
        (PopCntOp, i32),
    ],
)
def test_unary_traits(
    op_type: type[UnaryNumericalOperation] | type[EqzOp], operand_type: Attribute
):
    src = create_ssa_value(operand_type)

    _assert_traits(op_type(src), [Pure()], [Commutative()])


@pytest.mark.parametrize(
    "op_type, input_type, result_type, required_traits, forbidden_traits",
    [
        (ConvertSOp, i32, f32, [Pure()], [Commutative()]),
        (ConvertUOp, i64, f64, [Pure()], [Commutative()]),
        (DemoteOp, f64, f32, [Pure()], [Commutative()]),
        (PromoteOp, f32, f64, [Pure()], [Commutative()]),
        (WrapOp, i64, i32, [Pure()], [Commutative()]),
        (ReinterpretOp, i32, f32, [Pure()], [Commutative()]),
    ],
)
def test_conversion_traits(
    op_type: type[ConversionOperation],
    input_type: NumericType,
    result_type: NumericType,
    required_traits: list[OpTrait],
    forbidden_traits: list[OpTrait],
):
    input = create_ssa_value(input_type)

    _assert_traits(op_type(input, result_type), required_traits, forbidden_traits)


@pytest.mark.parametrize("op_type", [ExtendSI32Op, ExtendUI32Op])
def test_extend_i32_traits(op_type: type[ExtendSI32Op] | type[ExtendUI32Op]):
    input = create_ssa_value(i32)

    _assert_traits(op_type(input), [Pure()], [Commutative()])


@pytest.mark.parametrize("bits_to_take", [8, IntegerAttr(8, i64)])
def test_extend_low_bits_traits(bits_to_take: int | IntegerAttr):
    input = create_ssa_value(i32)

    _assert_traits(ExtendLowBitsSOp(input, bits_to_take), [Pure()], [Commutative()])
