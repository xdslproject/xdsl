import pytest

from xdsl.dialects.builtin import DYNAMIC_INDEX, i32
from xdsl.dialects.emitc import EmitC_ArrayType, EmitC_CallOpaqueOp, EmitC_ConstantOp
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


def test_constant_with_int():
    """
    Test that EmitC_ConstantOp can be created with a int value.
    """
    EmitC_ConstantOp(value=42)

