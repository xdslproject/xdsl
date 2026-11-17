import pytest

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    f32,
    i32,
    i64,
)
from xdsl.dialects.emitc import (
    EmitC_ArrayType,
    EmitC_CallOpaqueOp,
    EmitC_ConstantOp,
    EmitC_LiteralOp,
    EmitC_LValueType,
    EmitC_OpaqueAttr,
    EmitC_OpaqueType,
    EmitC_PtrDiffT,
    EmitC_SignedSizeT,
    EmitC_SizeT,
    verify_initialization_attribute,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute
from xdsl.traits import ConstantLike, is_side_effect_free
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        ValueError, match="expected static shape, but got dynamic dimension"
    ):
        EmitC_ArrayType([DYNAMIC_INDEX], i32)


def test_call_opaque_with_str_callee():
    """
    Test that EmitC_CallOpaqueOp can be created with a string callee.
    """
    EmitC_CallOpaqueOp(callee="test", call_args=[], result_types=[])


def test_constant_op_init():
    op = EmitC_ConstantOp(IntegerAttr(42, 32))

    op.verify()
    assert op.value == IntegerAttr(42, 32)
    assert op.result.type == i32

    assert EmitC_ConstantOp(IntegerAttr(42, 32), i32).is_structurally_equivalent(op)


def test_constant_op_float_init():
    op = EmitC_ConstantOp(FloatAttr(1.5, f32))

    op.verify()
    assert op.result.type == f32


def test_constant_op_opaque_init():
    value = EmitC_OpaqueAttr(StringAttr("42"))
    result_type = EmitC_OpaqueType(StringAttr("int"))
    op = EmitC_ConstantOp(value, result_type)

    op.verify()
    assert op.value == value
    assert op.result.type == result_type


def test_constant_op_opaque_requires_result_type():
    with pytest.raises(ValueError, match="result_type must be specified"):
        EmitC_ConstantOp(EmitC_OpaqueAttr(StringAttr("42")))


def test_constant_op_fold():
    value = IntegerAttr(42, 32)
    op = EmitC_ConstantOp(value)

    assert op.fold() == (value,)
    assert ConstantLike.get_constant_value(op.result) == value


@pytest.mark.parametrize(
    "result_type", [i32, EmitC_LValueType(i32), EmitC_OpaqueType(StringAttr("int"))]
)
def test_verify_initialization_attribute_opaque(result_type: Attribute):
    op = TestOp(result_types=[result_type])

    verify_initialization_attribute(op, EmitC_OpaqueAttr(StringAttr("42")))


def test_verify_initialization_attribute_matching_type():
    op = TestOp(result_types=[i32])

    verify_initialization_attribute(op, IntegerAttr(42, i32))


def test_verify_initialization_attribute_mismatched_type():
    op = TestOp(result_types=[i32])

    with pytest.raises(
        VerifyException,
        match=r"it's type \(i64\) to match the op's result type \(i32\)",
    ):
        verify_initialization_attribute(op, IntegerAttr(42, i64))


def test_verify_initialization_attribute_lvalue():
    op = TestOp(result_types=[EmitC_LValueType(i32)])

    verify_initialization_attribute(op, IntegerAttr(42, i32))

    with pytest.raises(
        VerifyException,
        match=r"it's type \(i64\) to match the op's result type \(i32\)",
    ):
        verify_initialization_attribute(op, IntegerAttr(42, i64))


@pytest.mark.parametrize(
    "result_type", [EmitC_PtrDiffT(), EmitC_SignedSizeT(), EmitC_SizeT()]
)
def test_verify_initialization_attribute_pointer_wide(result_type: Attribute):
    op = TestOp(result_types=[result_type])

    verify_initialization_attribute(op, IntegerAttr.from_index_int_value(42))

    with pytest.raises(VerifyException, match=r"it's type \(i32\) to match"):
        verify_initialization_attribute(op, IntegerAttr(42, i32))


def test_verify_initialization_attribute_index_requires_pointer_wide():
    op = TestOp(result_types=[i32])

    with pytest.raises(
        VerifyException,
        match=r"it's type \(index\) to match the op's result type \(i32\)",
    ):
        verify_initialization_attribute(op, IntegerAttr.from_index_int_value(42))


def test_literal_op_init():
    op = EmitC_LiteralOp("M_PI", f32)

    op.verify()
    assert op.value == StringAttr("M_PI")
    assert op.result.type == f32

    assert EmitC_LiteralOp(StringAttr("M_PI"), f32).is_structurally_equivalent(op)


def test_literal_op_is_side_effect_free():
    op = EmitC_LiteralOp("M_PI", f32)

    assert is_side_effect_free(op)
