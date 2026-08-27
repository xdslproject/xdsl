from collections.abc import Callable
from typing import Any

import pytest
from typing_extensions import TypeForm

from xdsl.jit.c_type_context import CFuncType
from xdsl.jit.py_type_context import PyTypeContext, TypeMap
from xdsl.utils.exceptions import JITException

FLOAT_MAP = TypeMap(float, "double")
INT_MAP = TypeMap(int, "int32_t")
BOOL_MAP = TypeMap(bool, "_Bool")


@pytest.fixture
def ctx() -> PyTypeContext:
    c = PyTypeContext()
    for type_map in (FLOAT_MAP, INT_MAP, BOOL_MAP):
        c.register_type_map(type_map)
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
            CFuncType(("double", "int32_t"), "_Bool"),
        ),
        (
            Callable[[float, float], bool],
            CFuncType(("double", "double"), "_Bool"),
        ),
        (Callable[[], float], CFuncType((), "double")),
    ],
    ids=["distinct-args", "repeated-args", "no-args"],
)
def test_c_func_type(
    ctx: PyTypeContext,
    signature: TypeForm[Callable[..., Any]],
    expected: CFuncType,
):
    assert ctx.func_type_map(signature).c_func_type() == expected


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
