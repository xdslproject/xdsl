from __future__ import annotations

import xdsl.dialects.builtin as builtin

from typing import Any, Callable, Generic, TypeAlias, TypeVar, Literal
from xdsl.dialects.builtin import Signedness


class _FrontendType:
    """Represents any type in the frontend."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        raise NotImplementedError()


# Type parameters for integers.
_Width = TypeVar("_Width", bound=int, covariant=True)
_Signedness = TypeVar("_Signedness", bound=Signedness, covariant=True)


class _Integer(Generic[_Width, _Signedness], _FrontendType):
    """Represents an integer type in the frontend. Should not be used explicitly."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.IntegerType.from_width


# Type aliases for signless integers.
i1: TypeAlias = _Integer[Literal[1], Literal[Signedness.SIGNLESS]]
i32: TypeAlias = _Integer[Literal[32], Literal[Signedness.SIGNLESS]]
i64: TypeAlias = _Integer[Literal[64], Literal[Signedness.SIGNLESS]]

# Type aliases for signed integers.
si32: TypeAlias = _Integer[Literal[32], Literal[Signedness.SIGNED]]
si64: TypeAlias = _Integer[Literal[64], Literal[Signedness.SIGNED]]

# Type aliases for unsigned integers.
ui32: TypeAlias = _Integer[Literal[32], Literal[Signedness.UNSIGNED]]
ui64: TypeAlias = _Integer[Literal[64], Literal[Signedness.UNSIGNED]]


class _Index(_FrontendType):
    """Represents an index type in the frontend. Should not be used explicitly."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.IndexType


# Type alias for index type.
index: TypeAlias = _Index


class _Float16(_FrontendType):
    """Represents a 16-bit floating-point type in the frontend. Should not be used explicitly."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float16Type


class _Float32(_FrontendType):
    """Represents a 32-bit floating-point type in the frontend. Should not be used explicitly."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float32Type


class _Float64(_FrontendType):
    """Represents a 64-bit floating-point type in the frontend. Should not be used explicitly."""

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float64Type


# Type alias for floating-point types.
f16: TypeAlias = _Float16
f32: TypeAlias = _Float32
f64: TypeAlias = _Float64
