import pytest

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    f32,
    i32,
)
from xdsl.dialects.emitc import (
    EmitC_ArrayType,
    EmitC_CallOpaqueOp,
    EmitC_ConstantOp,
    EmitC_LiteralOp,
    EmitC_OpaqueAttr,
    EmitC_OpaqueType,
)
from xdsl.traits import ConstantLike, is_side_effect_free


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


def test_literal_op_init():
    op = EmitC_LiteralOp("M_PI", f32)

    op.verify()
    assert op.value == StringAttr("M_PI")
    assert op.result.type == f32

    assert EmitC_LiteralOp(StringAttr("M_PI"), f32).is_structurally_equivalent(op)


def test_literal_op_is_side_effect_free():
    op = EmitC_LiteralOp("M_PI", f32)

    assert is_side_effect_free(op)
