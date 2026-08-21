import ctypes
from collections.abc import Callable

import pytest

from xdsl.jit.function import RawJITFunc, wrap_jit_func
from xdsl.jit.py_type_context import PyTypeContext, TypeMap
from xdsl.utils.exceptions import JITException


def add(lhs: int, rhs: int, /) -> int:
    return lhs + rhs


@pytest.fixture
def int_type_context() -> PyTypeContext:
    context = PyTypeContext()
    context.register_type_map(TypeMap(int, ctypes.c_int64, ctypes.c_int64, int))
    return context


@pytest.fixture
def raw_add_func() -> RawJITFunc:
    c_func_type = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    return RawJITFunc(c_func_type, c_func_type(add))


def test_wrap_jit_func_marshals_arguments_and_result():
    # record conversions to check their order
    conversions: list[tuple[str, int]] = []

    def to_ctype(value: int) -> ctypes.c_int64:
        conversions.append(("to", value))
        return ctypes.c_int64(value + 1)

    def from_ctype(value: int) -> int:
        conversions.append(("from", value))
        return value * 10

    py_type_context = PyTypeContext()
    py_type_context.register_type_map(
        TypeMap(int, ctypes.c_int64, to_ctype, from_ctype)
    )

    # differ from `add` to prove this native function is called
    def native_func(lhs: int, rhs: int) -> int:
        return lhs - rhs

    c_func_type = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    raw_func = RawJITFunc(c_func_type, c_func_type(native_func))
    wrapped_func = wrap_jit_func(
        raw_func,
        add,
        Callable[[int, int], int],
        py_type_context,
    )

    assert wrapped_func(1, 2) == -10
    assert conversions == [("to", 1), ("to", 2), ("from", -1)]
    assert wrapped_func.original_func is add
    assert wrapped_func.raw_func is raw_func


def test_wrap_jit_func_rejects_mismatched_c_signature():
    py_type_context = PyTypeContext()
    py_type_context.register_type_map(TypeMap(int, ctypes.c_int64, ctypes.c_int64, int))

    def original_func(value: int, /) -> int:
        return value

    def native_func(value: float) -> float:
        return value

    c_func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    raw_func = RawJITFunc(c_func_type, c_func_type(native_func))

    with pytest.raises(JITException, match="does not match signature"):
        wrap_jit_func(
            raw_func,
            original_func,
            Callable[[int], int],
            py_type_context,
        )


def test_wrapped_jit_func_rejects_keyword_arguments(
    raw_add_func: RawJITFunc, int_type_context: PyTypeContext
):
    wrapped_func = wrap_jit_func(
        raw_add_func,
        add,
        Callable[[int, int], int],
        int_type_context,
    )

    def call_with_keywords(func: Callable[..., int]) -> int:
        return func(lhs=1, rhs=2)

    with pytest.raises(JITException, match="do not support keyword arguments"):
        call_with_keywords(wrapped_func)
