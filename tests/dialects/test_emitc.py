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
    EmitC_LoadOp,
    EmitC_LValueType,
    EmitC_OpaqueAttr,
    EmitC_OpaqueType,
    EmitC_PtrDiffT,
    EmitC_SignedSizeT,
    EmitC_SizeT,
    EmitC_VariableOp,
    verify_initialization_attribute,
)
from xdsl.ir import Attribute
from xdsl.traits import ConstantLike, is_side_effect_free
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


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
    op = EmitC_ConstantOp(IntegerAttr(42, 32), i32)

    op.verify()
    assert op.value == IntegerAttr(42, 32)
    assert op.result.type == i32


def test_constant_op_float_init():
    op = EmitC_ConstantOp(FloatAttr(1.5, f32), f32)

    op.verify()
    assert op.value == FloatAttr(1.5, f32)
    assert op.result.type == f32


def test_constant_op_opaque_init():
    value = EmitC_OpaqueAttr(StringAttr("42"))
    result_type = EmitC_OpaqueType(StringAttr("int"))
    op = EmitC_ConstantOp(value, result_type)

    op.verify()
    assert op.value == value
    assert op.result.type == result_type


def test_constant_op_fold():
    value = IntegerAttr(42, 32)
    op = EmitC_ConstantOp(value, i32)

    assert op.fold() == (value,)
    assert ConstantLike.get_constant_value(op.result) == value


@pytest.mark.parametrize(
    "result_type", [i32, EmitC_LValueType(i32), EmitC_OpaqueType(StringAttr("int"))]
)
def test_verify_initialization_attribute_opaque(result_type: Attribute):
    verify_initialization_attribute(result_type, EmitC_OpaqueAttr(StringAttr("42")))


def test_verify_initialization_attribute_matching_type():
    verify_initialization_attribute(i32, IntegerAttr(42, i32))


def test_verify_initialization_attribute_mismatched_type():
    with pytest.raises(
        VerifyException,
        match=r"its type \(i64\) to match the op's result type \(i32\)",
    ):
        verify_initialization_attribute(i32, IntegerAttr(42, i64))


def test_verify_initialization_attribute_lvalue():
    result_type = EmitC_LValueType(i32)

    verify_initialization_attribute(result_type, IntegerAttr(42, i32))

    with pytest.raises(
        VerifyException,
        match=r"its type \(i64\) to match the op's result type \(i32\)",
    ):
        verify_initialization_attribute(result_type, IntegerAttr(42, i64))


@pytest.mark.parametrize(
    "result_type", [EmitC_PtrDiffT(), EmitC_SignedSizeT(), EmitC_SizeT()]
)
def test_verify_initialization_attribute_pointer_wide(result_type: Attribute):
    verify_initialization_attribute(result_type, IntegerAttr.from_index_int_value(42))

    with pytest.raises(VerifyException, match=r"its type \(i32\) to match"):
        verify_initialization_attribute(result_type, IntegerAttr(42, i32))


def test_verify_initialization_attribute_index_requires_pointer_wide():
    with pytest.raises(
        VerifyException,
        match=r"its type \(index\) to match the op's result type \(i32\)",
    ):
        verify_initialization_attribute(i32, IntegerAttr.from_index_int_value(42))


def test_literal_op_init():
    op = EmitC_LiteralOp("M_PI", f32)

    op.verify()
    assert op.value == StringAttr("M_PI")
    assert op.result.type == f32

    assert EmitC_LiteralOp(StringAttr("M_PI"), f32).is_structurally_equivalent(op)


def test_literal_op_is_side_effect_free():
    op = EmitC_LiteralOp("M_PI", f32)

    assert is_side_effect_free(op)


def test_load_op_init():
    lvalue = create_ssa_value(EmitC_LValueType(i32))
    op = EmitC_LoadOp(lvalue, i32)

    op.verify()
    assert op.operand is lvalue
    assert op.result.type == i32


def test_load_op_init_from_operation():
    lvalue = create_ssa_value(EmitC_LValueType(i32))
    op = EmitC_LoadOp(lvalue.owner, i32)

    op.verify()
    assert op.operand is lvalue
    assert op.result.type == i32


def test_variable_op_init():
    op = EmitC_VariableOp(IntegerAttr(7, 32), EmitC_LValueType(i32))

    op.verify()
    assert op.value == IntegerAttr(7, 32)
    assert op.result.type == EmitC_LValueType(i32)


def test_variable_op_opaque_init():
    value = EmitC_OpaqueAttr(StringAttr(""))
    result_type = EmitC_ArrayType([2], i32)
    op = EmitC_VariableOp(value, result_type)

    op.verify()
    assert op.value == value
    assert op.result.type == result_type
