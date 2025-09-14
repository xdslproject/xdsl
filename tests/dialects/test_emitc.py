import pytest

from xdsl.dialects.builtin import i32
from xdsl.dialects.emitc import EmitC_ArrayType, EmitC_CallOpaqueOp
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="expected static shape, but got dynamic dimension"
    ):
        EmitC_ArrayType([-1], i32)


def test_call_opaque_with_str_callee():
    """
    Test that EmitC_CallOpaqueOp can be created with a string callee.
    """
    EmitC_CallOpaqueOp(callee="test", call_args=[], result_types=[])
