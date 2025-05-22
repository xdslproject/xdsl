import pytest

from xdsl.dialects.builtin import MemRefType, i32
from xdsl.dialects.emitc import EmitC_ArrayType
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_empty_shape():
    with pytest.raises(VerifyException, match="EmitC array shape must not be empty"):
        EmitC_ArrayType([], i32)


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="EmitC array dimensions must have non-negative size"
    ):
        EmitC_ArrayType([-1], i32)


def test_emitc_array_nested_array_type():
    nested_array = EmitC_ArrayType([2], i32)
    with pytest.raises(
        VerifyException,
        match="EmitC array element type cannot be another EmitC_ArrayType.",
    ):
        EmitC_ArrayType([1], nested_array)


def test_emitc_array_unsupported_element_type():
    unsupported_type = MemRefType(element_type=i32, shape=[1])

    with pytest.raises(
        VerifyException,
        match=f"EmitC array element type '{unsupported_type}' is not a supported EmitC type.",
    ):
        EmitC_ArrayType([1], unsupported_type)
