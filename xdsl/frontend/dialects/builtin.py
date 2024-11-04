from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Literal, TypeAlias, TypeVar

import xdsl.dialects.builtin as builtin
from xdsl.dialects.builtin import Signedness


class _FrontendType:
    """Represents any type in the frontend."""

    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        raise NotImplementedError()


# Type parameters for integers.
_Width = TypeVar("_Width", bound=int)
_Signedness = TypeVar("_Signedness", bound=Signedness)


# Note the types ignored below:
# a) on each function, since the functions are constrained on a limited set of
#    known types, and _Integer can represent types outside of that set.
# b) on functions that return `bool` in object, instead of `i1`
class _Integer(Generic[_Width, _Signedness], _FrontendType):
    """
    Represents an integer type in the frontend. Should not be used explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return builtin.IntegerType.__call__

    def __add__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import addi

        return addi(
            self,  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
        )

    def __and__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import andi

        return andi(
            self,  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
        )

    def __lshift__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import shli

        return shli(
            self,  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
        )

    def __mul__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import muli

        return muli(
            self,  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
        )

    def __rshift__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import shrsi

        return shrsi(
            self,  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
        )

    def __sub__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import subi

        return subi(
            self,  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
        )

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: _Integer[_Width, _Signedness]
    ) -> i1:
        from xdsl.frontend.dialects.arith import cmpi

        return cmpi(
            self,  # pyright: ignore[reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
            "eq",
        )

    def __ge__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi

        return cmpi(
            self,  # pyright: ignore[reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
            "sge",
        )

    def __gt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi

        return cmpi(
            self,  # pyright: ignore[reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
            "sgt",
        )

    def __le__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi

        return cmpi(
            self,  # pyright: ignore[reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
            "sle",
        )

    def __lt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi

        return cmpi(
            self,  # pyright: ignore[reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
            "slt",
        )

    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: _Integer[_Width, _Signedness]
    ) -> i1:
        from xdsl.frontend.dialects.arith import cmpi

        return cmpi(
            self,  # pyright: ignore[reportArgumentType]
            other,  # pyright: ignore[reportArgumentType]
            "ne",
        )


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
    """
    Represents an index type in the frontend. Should not be used explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return builtin.IndexType.__call__


# Type alias for index type.
index: TypeAlias = _Index


class _Float16(_FrontendType):
    """
    Represents a 16-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return builtin.Float16Type.__call__

    def __add__(self, other: f16) -> f16:
        from xdsl.frontend.dialects.arith import addf

        return addf(self, other)

    def __sub__(self, other: f16) -> f16:
        from xdsl.frontend.dialects.arith import subf

        return subf(self, other)

    def __mul__(self, other: f16) -> f16:
        from xdsl.frontend.dialects.arith import mulf

        return mulf(self, other)


class _Float32(_FrontendType):
    """
    Represents a 32-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return builtin.Float32Type.__call__

    def __add__(self, other: f32) -> f32:
        from xdsl.frontend.dialects.arith import addf

        return addf(self, other)

    def __sub__(self, other: f32) -> f32:
        from xdsl.frontend.dialects.arith import subf

        return subf(self, other)

    def __mul__(self, other: f32) -> f32:
        from xdsl.frontend.dialects.arith import mulf

        return mulf(self, other)


class _Float64(_FrontendType):
    """
    Represents a 64-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[[], Any]:
        return builtin.Float64Type.__call__

    def __add__(self, other: f64) -> f64:
        from xdsl.frontend.dialects.arith import addf

        return addf(self, other)

    def __sub__(self, other: f64) -> f64:
        from xdsl.frontend.dialects.arith import subf

        return subf(self, other)

    def __mul__(self, other: f64) -> f64:
        from xdsl.frontend.dialects.arith import mulf

        return mulf(self, other)


# Type alias for floating-point types.
f16: TypeAlias = _Float16
f32: TypeAlias = _Float32
f64: TypeAlias = _Float64
