import pytest

from xdsl.dialects.builtin import i32
from xdsl.dialects.emitc import EmitC_ArrayType
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_empty_shape():
    with pytest.raises(VerifyException, match="EmitC array shape must not be empty"):
        EmitC_ArrayType([], i32)
