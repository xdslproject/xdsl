from __future__ import annotations

import ast
from collections.abc import Callable
from sys import _getframe  # pyright: ignore[reportPrivateUsage]
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import pytest

from xdsl.frontend.pyast.dialects.builtin import (
    _FrontendType,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.type_conversion import TypeConverter
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class A(ParametrizedAttribute):
    name = "a"


class _A(_FrontendType):
    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return A.__call__


class _B:
    pass


T = TypeVar("T")


class _C(Generic[T]):
    pass


@irdl_attr_definition
class D(ParametrizedAttribute):
    name = "d"


class _D(Generic[T], _FrontendType):
    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return D.__call__


a: TypeAlias = _A
b: TypeAlias = _B
c2: TypeAlias = _C[Literal[2]]
d12: TypeAlias = _D[tuple[Literal[1], Literal[2]]]

globals: dict[str, Any] = _getframe(0).f_globals


def test_raises_exception_on_unknown_type():
    type_converter = TypeConverter(globals)
    type_hint = ast.Name("unknown", lineno=0, col_offset=0)

    with pytest.raises(CodeGenerationException) as err:
        type_converter.convert_type_hint(type_hint)
    assert err.value.msg == "Unknown type hint 'unknown'."


def test_raises_exception_on_non_frontend_type_I():
    type_converter = TypeConverter(globals)
    type_hint = ast.Name("b", lineno=0, col_offset=0)

    with pytest.raises(CodeGenerationException) as err:
        type_converter.convert_type_hint(type_hint)
    assert (
        err.value.msg == "Unknown type hint for type 'b' inside 'ast.Name' expression."
    )


def test_raises_exception_on_non_frontend_type_II():
    type_converter = TypeConverter(globals)
    type_hint = ast.Name("c2", lineno=0, col_offset=0)

    with pytest.raises(CodeGenerationException) as err:
        type_converter.convert_type_hint(type_hint)
    assert err.value.msg == "'c2' is not a frontend type."


def test_raises_exception_on_nontrivial_generics():
    type_converter = TypeConverter(globals)
    type_hint = ast.Name("d12", lineno=0, col_offset=0)

    with pytest.raises(CodeGenerationException) as err:
        type_converter.convert_type_hint(type_hint)
    assert (
        err.value.msg
        == "Expected 1 type argument for generic type 'd12', got 2 type arguments instead."
    )


def test_type_conversion_caches_type():
    type_converter = TypeConverter(globals)
    type_hint = ast.Name("a", lineno=0, col_offset=0)

    assert "a" not in type_converter.name_to_xdsl_type_map
    xdsl_type = type_converter.convert_type_hint(type_hint)
    assert type_converter.name_to_xdsl_type_map["a"] == xdsl_type
    assert type_converter.xdsl_to_frontend_type_map[xdsl_type.__class__] == _A
