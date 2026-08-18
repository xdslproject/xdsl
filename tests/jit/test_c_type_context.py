import ctypes

import pytest

from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    IndexType,
    IntegerType,
    NoneType,
    Signedness,
)
from xdsl.ir import Attribute
from xdsl.jit.c_type_context import CTypeContext, register_builtin_ctypes
from xdsl.utils.exceptions import JITException


@pytest.fixture
def ctx() -> CTypeContext:
    c = CTypeContext()
    register_builtin_ctypes(c)
    return c


@pytest.mark.parametrize(
    "type_attr, expected",
    [
        (IntegerType(1), ctypes.c_bool),
        (IntegerType(8), ctypes.c_int8),
        (IntegerType(16), ctypes.c_int16),
        (IntegerType(32), ctypes.c_int32),
        (IntegerType(64), ctypes.c_int64),
        (IntegerType(32, Signedness.SIGNED), ctypes.c_int32),
        (IntegerType(32, Signedness.UNSIGNED), ctypes.c_int32),
        (Float32Type(), ctypes.c_float),
        (Float64Type(), ctypes.c_double),
        (NoneType(), None),
    ],
)
def test_builtin_resolve(ctx: CTypeContext, type_attr: Attribute, expected: object):
    assert ctx.to_ctype(type_attr) is expected


@pytest.mark.parametrize(
    "type_attr, match",
    [
        (IntegerType(0), "integer of width 0"),
        (IntegerType(17), "integer of width 17"),
        (IntegerType(128), "integer of width 128"),
        (Float16Type(), "No ctypes mapping for type"),
        (IndexType(), "No ctypes mapping for type"),
    ],
)
def test_builtin_unsupported(ctx: CTypeContext, type_attr: Attribute, match: str):
    with pytest.raises(JITException, match=match):
        ctx.to_ctype(type_attr)


def test_to_c_func_type(ctx: CTypeContext):
    assert ctx.to_c_func_type(
        (Float64Type(), IntegerType(32)), Float64Type()
    ) is ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int32)


def test_to_c_func_type_propagates_unmapped_type(ctx: CTypeContext):
    with pytest.raises(JITException, match="No ctypes mapping for type: index"):
        ctx.to_c_func_type((IndexType(),), Float64Type())
