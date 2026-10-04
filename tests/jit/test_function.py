from collections.abc import Callable

import pytest

from xdsl.jit.c_type_context import CFuncSignature
from xdsl.jit.function import RawJITFunc, wrap_jit_func
from xdsl.jit.py_type_context import PyTypeContext, TypeMap
from xdsl.utils.exceptions import JITException


def add(lhs: int, rhs: int, /) -> int:
    return lhs + rhs


def test_wrap_jit_func_converts_arguments_and_result():
    # record conversions to check their order
    conversions: list[tuple[str, int]] = []

    class FalseyConverter:
        def __bool__(self) -> bool:
            return False

        def __call__(self, value: int) -> int:
            conversions.append(("to", value))
            return value + 1

    def from_c(value: int) -> int:
        conversions.append(("from", value))
        return value * 10

    py_type_context = PyTypeContext()
    py_type_context.register_type_map(
        TypeMap(int, "int64_t", FalseyConverter(), from_c)
    )

    def subtract(lhs: int, rhs: int) -> int:
        return lhs - rhs

    c_func_type = CFuncSignature(("int64_t", "int64_t"), "int64_t")
    raw_jit_func = RawJITFunc(c_func_type, subtract)
    wrapped_func = wrap_jit_func(
        raw_jit_func,
        add,
        Callable[[int, int], int],
        py_type_context,
    )

    assert wrapped_func(1, 2) == -10
    assert conversions == [("to", 1), ("to", 2), ("from", -1)]
    assert wrapped_func.original_func is add
    assert wrapped_func.raw_func is raw_jit_func


def test_wrap_jit_func_rejects_mismatched_c_signature():
    py_type_context = PyTypeContext()
    py_type_context.register_type_map(TypeMap(int, "int64_t"))

    def original_func(value: int, /) -> int:
        return value

    def native_func(value: float) -> float:
        return value

    raw_func = RawJITFunc(CFuncSignature(("double",), "double"), native_func)

    with pytest.raises(JITException, match="does not match signature"):
        wrap_jit_func(
            raw_func,
            original_func,
            Callable[[int], int],
            py_type_context,
        )
