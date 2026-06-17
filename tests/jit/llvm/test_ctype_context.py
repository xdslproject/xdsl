import ctypes

import pytest

from xdsl.dialects.builtin import (
    Float16Type,
    Float32Type,
    Float64Type,
    IndexType,
    IntAttr,
    IntegerType,
    NoneType,
    Signedness,
)
from xdsl.dialects.llvm import LLVMPointerType, LLVMVoidType
from xdsl.ir import Attribute
from xdsl.jit.llvm.ctype_context import CTypeContext, register_builtin_ctypes
from xdsl.utils.exceptions import LLVMTranslationException


def test_register_and_resolve():
    ctx = CTypeContext()
    ctx.register_ctype(Float32Type, lambda _: ctypes.c_float)
    assert ctx.to_ctype(Float32Type()) is ctypes.c_float


def test_converter_receives_the_attribute():
    ctx = CTypeContext()
    seen: list[IntegerType] = []

    def converter(t: IntegerType):
        seen.append(t)
        return ctypes.c_int32

    ctx.register_ctype(IntegerType, converter)
    int_t = IntegerType(32)
    assert ctx.to_ctype(int_t) is ctypes.c_int32
    assert seen == [int_t]


def test_resolve_unregistered_raises():
    ctx = CTypeContext()
    with pytest.raises(LLVMTranslationException, match="No ctypes mapping"):
        ctx.to_ctype(Float32Type())


def test_re_register_overwrites():
    ctx = CTypeContext()
    ctx.register_ctype(Float32Type, lambda _: ctypes.c_float)
    ctx.register_ctype(Float32Type, lambda _: ctypes.c_double)
    assert ctx.to_ctype(Float32Type()) is ctypes.c_double


def test_two_contexts_are_independent():
    a = CTypeContext()
    b = CTypeContext()
    a.register_ctype(Float32Type, lambda _: ctypes.c_float)
    with pytest.raises(LLVMTranslationException):
        b.to_ctype(Float32Type())


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
        (LLVMPointerType(), ctypes.c_void_p),
        (LLVMPointerType(IntAttr(1)), ctypes.c_void_p),
        (LLVMVoidType(), None),
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
    with pytest.raises(LLVMTranslationException, match=match):
        ctx.to_ctype(type_attr)
