import ctypes
from collections.abc import Callable
from typing import Any

import pytest
from typing_extensions import TypeForm

from xdsl.jit.py_type_context import PyTypeContext, TypeMap
from xdsl.utils.exceptions import JITException


def _convert(_: Any) -> Any: ...


FLOAT_MAP = TypeMap(float, ctypes.c_double, _convert, _convert)
INT_MAP = TypeMap(int, ctypes.c_int32, _convert, _convert)
BOOL_MAP = TypeMap(bool, ctypes.c_bool, _convert, _convert)


@pytest.fixture
def ctx() -> PyTypeContext:
    c = PyTypeContext()
    for type_map in (FLOAT_MAP, INT_MAP, BOOL_MAP):
        c.extend(type_map)
    return c


def test_type_map_is_per_context(ctx: PyTypeContext):
    assert ctx.type_map(float) is FLOAT_MAP
    with pytest.raises(JITException, match="<class 'float'>"):
        PyTypeContext().type_map(float)


def test_func_type_map(ctx: PyTypeContext):
    func_type_map = ctx.func_type_map(Callable[[float, int], bool])
    assert func_type_map.arg_maps == (FLOAT_MAP, INT_MAP)
    assert func_type_map.res_map is BOOL_MAP


@pytest.mark.parametrize(
    "signature, expected",
    [
        (
            Callable[[float, int], bool],
            (ctypes.c_bool, ctypes.c_double, ctypes.c_int32),
        ),
        (
            Callable[[float, float], bool],
            (ctypes.c_bool, ctypes.c_double, ctypes.c_double),
        ),
        (Callable[[], float], (ctypes.c_double,)),
    ],
    ids=["distinct-args", "repeated-args", "no-args"],
)
def test_c_func_type(
    ctx: PyTypeContext,
    signature: TypeForm[Callable[..., Any]],
    expected: tuple[type[Any], ...],
):
    assert ctx.func_type_map(signature).c_func_type() is ctypes.CFUNCTYPE(*expected)


@pytest.mark.parametrize(
    "signature",
    [Callable[[str], float], Callable[[float], str]],
    ids=["argument", "result"],
)
def test_unregistered_type_raises(
    ctx: PyTypeContext, signature: TypeForm[Callable[..., Any]]
):
    with pytest.raises(
        JITException, match="No type map for Python type: <class 'str'>"
    ):
        ctx.func_type_map(signature)


@pytest.mark.parametrize(
    "signature", [Callable, Callable[..., float]], ids=["bare", "ellipsis"]
)
def test_unenumerable_signature_raises(signature: TypeForm[Callable[..., Any]]):
    with pytest.raises(JITException, match="Unsupported signature"):
        PyTypeContext().func_type_map(signature)
