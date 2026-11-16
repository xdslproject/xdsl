import pytest

from xdsl.dialects.builtin import DYNAMIC_INDEX, StringAttr, f32, i32
from xdsl.dialects.emitc import EmitC_ArrayType, EmitC_CallOpaqueOp, EmitC_LiteralOp
from xdsl.traits import is_side_effect_free


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


def test_literal_op_init():
    op = EmitC_LiteralOp("M_PI", f32)

    op.verify()
    assert op.value == StringAttr("M_PI")
    assert op.result.type == f32

    assert EmitC_LiteralOp(StringAttr("M_PI"), f32).is_structurally_equivalent(op)


def test_literal_op_is_side_effect_free():
    op = EmitC_LiteralOp("M_PI", f32)

    assert is_side_effect_free(op)
