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
    """
    Represents an integer type in the frontend. Should not be used explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.IntegerType.from_width

    def __add__(
            self,
            other: _Integer[_Width,
                            _Signedness]) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import addi
        return addi(self, other)

    def __and__(
            self,
            other: _Integer[_Width,
                            _Signedness]) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import andi
        return andi(self, other)

    def __lshift__(
            self,
            other: _Integer[_Width,
                            _Signedness]) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import shli
        return shli(self, other)

    def __mul__(
            self,
            other: _Integer[_Width,
                            _Signedness]) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import muli
        return muli(self, other)

    def __rshift__(
            self,
            other: _Integer[_Width,
                            _Signedness]) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import shrsi
        return shrsi(self, other)

    def __sub__(
            self,
            other: _Integer[_Width,
                            _Signedness]) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.dialects.arith import subi
        return subi(self, other)

    def __eq__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "eq")

    def __ge__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "sge")

    def __gt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "sgt")

    def __le__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "sle")

    def __lt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "slt")

    def __ne__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "ne")


# Type aliases for signless integers.
i1: TypeAlias = _Integer[Literal[1], Literal[Signedness.SIGNLESS]]
i8: TypeAlias = _Integer[Literal[8], Literal[Signedness.SIGNLESS]]
i16: TypeAlias = _Integer[Literal[16], Literal[Signedness.SIGNLESS]]
i32: TypeAlias = _Integer[Literal[32], Literal[Signedness.SIGNLESS]]
i64: TypeAlias = _Integer[Literal[64], Literal[Signedness.SIGNLESS]]
i128: TypeAlias = _Integer[Literal[128], Literal[Signedness.SIGNLESS]]
i255: TypeAlias = _Integer[Literal[255], Literal[Signedness.SIGNLESS]]

# Type aliases for signed integers.
si1: TypeAlias = _Integer[Literal[1], Literal[Signedness.SIGNED]]
si8: TypeAlias = _Integer[Literal[8], Literal[Signedness.SIGNED]]
si16: TypeAlias = _Integer[Literal[16], Literal[Signedness.SIGNED]]
si32: TypeAlias = _Integer[Literal[32], Literal[Signedness.SIGNED]]
si64: TypeAlias = _Integer[Literal[64], Literal[Signedness.SIGNED]]
si128: TypeAlias = _Integer[Literal[128], Literal[Signedness.SIGNED]]
si255: TypeAlias = _Integer[Literal[255], Literal[Signedness.SIGNED]]

# Type aliases for unsigned integers.
ui1: TypeAlias = _Integer[Literal[1], Literal[Signedness.UNSIGNED]]
ui8: TypeAlias = _Integer[Literal[8], Literal[Signedness.UNSIGNED]]
ui16: TypeAlias = _Integer[Literal[16], Literal[Signedness.UNSIGNED]]
ui32: TypeAlias = _Integer[Literal[32], Literal[Signedness.UNSIGNED]]
ui64: TypeAlias = _Integer[Literal[64], Literal[Signedness.UNSIGNED]]
ui128: TypeAlias = _Integer[Literal[128], Literal[Signedness.UNSIGNED]]
ui255: TypeAlias = _Integer[Literal[255], Literal[Signedness.UNSIGNED]]


class _Index(_FrontendType):
    """
    Represents an index type in the frontend. Should not be used explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.IndexType


# Type alias for index type.
index: TypeAlias = _Index


class _Float16(_FrontendType):
    """
    Represents a 16-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float16Type

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
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float32Type

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
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float64Type

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
