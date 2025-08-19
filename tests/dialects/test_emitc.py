import pytest

from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.emitc import EmitC_ArrayType, EmitC_CallOpaqueOp
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="EmitC array dimensions must have non-negative size"
    ):
        EmitC_ArrayType([-1], IntegerType(32))


def test_call_opaque_with_str_callee():
    """
    Test that EmitC_CallOpaqueOp can be created with a string callee.
    """
    EmitC_CallOpaqueOp(callee="test", call_args=[], result_types=[])
