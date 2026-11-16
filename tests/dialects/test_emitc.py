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
    EmitC_AssignOp,
    EmitC_CallOpaqueOp,
    EmitC_ConstantOp,
    EmitC_LiteralOp,
    EmitC_LoadOp,
    EmitC_LValueType,
    EmitC_OpaqueAttr,
    EmitC_OpaqueType,
    EmitC_VariableOp,
    verify_initialization_attribute,
)
from xdsl.traits import ConstantLike, is_side_effect_free
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="expected static shape, but got dynamic dimension"
    ):
        EmitC_ArrayType([DYNAMIC_INDEX], i32)


def test_call_opaque_with_str_callee():
    """
    Test that EmitC_CallOpaqueOp can be created with a string callee.
    """
    EmitC_CallOpaqueOp(callee="test", call_args=[], result_types=[])


def test_constant_op_integer_init():
    op = EmitC_ConstantOp(IntegerAttr(42, 32))

    assert op.value == IntegerAttr(42, 32)
    assert op.result.type == i32
    op.verify()


def test_constant_op_opaque_init():
    value = EmitC_OpaqueAttr(StringAttr("42"))
    result_type = EmitC_OpaqueType(StringAttr("int"))
    op = EmitC_ConstantOp(value, result_type)

    assert op.value == value
    assert op.result.type == result_type
    op.verify()


def test_constant_op_opaque_requires_result_type():
    with pytest.raises(ValueError, match="result_type must be specified"):
        EmitC_ConstantOp(EmitC_OpaqueAttr(StringAttr("42")))


def test_variable_op_init():
    op = EmitC_VariableOp(IntegerAttr(7, 32), EmitC_LValueType(i32))

    assert op.value == IntegerAttr(7, 32)
    assert op.result.type == EmitC_LValueType(i32)
    op.verify()


def test_load_op_infers_result_type():
    var = EmitC_VariableOp(EmitC_OpaqueAttr(StringAttr("")), EmitC_LValueType(i32))
    op = EmitC_LoadOp(var.result)

    assert op.operand.owner is var
    assert op.result.type == i32
    op.verify()


def test_assign_op_init():
    var = EmitC_VariableOp(EmitC_OpaqueAttr(StringAttr("")), EmitC_LValueType(i32))
    value = EmitC_ConstantOp(IntegerAttr(42, 32))
    op = EmitC_AssignOp(var.result, value.result)

    assert op.var.owner is var
    assert op.value.owner is value
    assert not op.results
    op.verify()


def test_constant_op_zero_is_valid():
    op = EmitC_ConstantOp(IntegerAttr(0, 32))
    op.verify()


def test_constant_op_float_init():
    op = EmitC_ConstantOp(FloatAttr(1.5, f32))

    assert op.result.type == f32
    op.verify()


def test_constant_op_empty_opaque_fails():
    op = EmitC_ConstantOp(
        EmitC_OpaqueAttr(StringAttr("")), EmitC_OpaqueType(StringAttr("int"))
    )
    with pytest.raises(VerifyException, match="value must not be empty"):
        op.verify()


def test_variable_op_zero_is_valid():
    op = EmitC_VariableOp(IntegerAttr(0, 32), EmitC_LValueType(i32))
    op.verify()


def test_variable_op_mismatched_value_type_fails():
    op = EmitC_VariableOp(IntegerAttr(42, 64), EmitC_LValueType(i32))
    with pytest.raises(VerifyException, match="to match the op's result type"):
        op.verify()


def test_assign_op_mismatched_types_fails():
    var = EmitC_VariableOp(EmitC_OpaqueAttr(StringAttr("")), EmitC_LValueType(i32))
    value = EmitC_ConstantOp(IntegerAttr(42, i64))
    op = EmitC_AssignOp(var.result, value.result)
    with pytest.raises(
        VerifyException, match=r"requires value's type \(i64\) to match variable's"
    ):
        op.verify()


def test_constant_op_fold():
    value = IntegerAttr(42, 32)
    op = EmitC_ConstantOp(value)

    assert op.fold() == (value,)
    assert ConstantLike.get_constant_value(op.result) == value


def test_literal_op_is_side_effect_free():
    op = EmitC_LiteralOp(properties={"value": StringAttr("M_PI")}, result_types=[f32])

    op.verify()
    assert is_side_effect_free(op)


def test_initialization_attribute_rejects_string_attr():
    op = EmitC_ConstantOp(IntegerAttr(42, 32))
    with pytest.raises(VerifyException, match="string attributes are not supported"):
        verify_initialization_attribute(op, StringAttr("42"))
