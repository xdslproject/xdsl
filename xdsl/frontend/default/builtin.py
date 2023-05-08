from __future__ import annotations
from typing import Union

import xdsl.dialects.builtin as builtin

from typing import Any, Callable, Generic, TypeAlias, TypeVar, Literal
from xdsl.dialects.builtin import Signedness
from xdsl.frontend.type import FrontendType

# Type parameters for integers.
_Width = TypeVar("_Width", bound=int)
_Signedness = TypeVar("_Signedness", bound=Signedness)


# Note the types ignored below:
# a) on each function, since the functions are constrained on a limited set of
#    known types, and IntegerType can represent types outside of that set.
# b) on functions that return `bool` in object, instead of `i1`
class IntegerType(Generic[_Width, _Signedness], FrontendType):
    """
    Represents an integer type in the frontend. Should not be used explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.IntegerType

    def __add__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import addi

        return addi(self, other)  # type: ignore

    def __and__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import andi

        return andi(self, other)  # type: ignore

    def __lshift__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import shli

        return shli(self, other)  # type: ignore

    def __mul__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import muli

        return muli(self, other)  # type: ignore

    def __rshift__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import shrsi

        return shrsi(self, other)  # type: ignore

    def __sub__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import subi

        return subi(self, other)  # type: ignore

    def __eq__(self, other: _Integer[_Width, _Signedness]) -> i1:  # type: ignore
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "eq")  # type: ignore

    def __ge__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "sge")  # type: ignore

    def __gt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "sgt")  # type: ignore

    def __le__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "sle")  # type: ignore

    def __lt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "slt")  # type: ignore

    def __ne__(self, other: _Integer[_Width, _Signedness]) -> i1:  # type: ignore
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "ne")  # type: ignore

    def __add__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import addi

        return addi(self, other)

    def __and__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import andi

        return andi(self, other)

    def __lshift__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import shli

        return shli(self, other)

    def __mul__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import muli

        return muli(self, other)

    def __rshift__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import shrsi

        return shrsi(self, other)

    def __sub__(
        self, other: _Integer[_Width, _Signedness]
    ) -> _Integer[_Width, _Signedness]:
        from xdsl.frontend.default.arith import subi

        return subi(self, other)

    def __eq__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "eq")

    def __ge__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "sge")

    def __gt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "sgt")

    def __le__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "sle")

    def __lt__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "slt")

    def __ne__(self, other: _Integer[_Width, _Signedness]) -> i1:
        from xdsl.frontend.default.arith import cmpi

        return cmpi(self, other, "ne")


# Type aliases for signless integers.
i1: TypeAlias = IntegerType[Literal[1], Literal[Signedness.SIGNLESS]]
i8: TypeAlias = IntegerType[Literal[8], Literal[Signedness.SIGNLESS]]
i16: TypeAlias = IntegerType[Literal[16], Literal[Signedness.SIGNLESS]]
i32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.SIGNLESS]]
i64: TypeAlias = IntegerType[Literal[64], Literal[Signedness.SIGNLESS]]
i128: TypeAlias = IntegerType[Literal[128], Literal[Signedness.SIGNLESS]]
i255: TypeAlias = IntegerType[Literal[255], Literal[Signedness.SIGNLESS]]

# Type aliases for signed integers.
si1: TypeAlias = IntegerType[Literal[1], Literal[Signedness.SIGNED]]
si8: TypeAlias = IntegerType[Literal[8], Literal[Signedness.SIGNED]]
si16: TypeAlias = IntegerType[Literal[16], Literal[Signedness.SIGNED]]
si32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.SIGNED]]
si64: TypeAlias = IntegerType[Literal[64], Literal[Signedness.SIGNED]]
si128: TypeAlias = IntegerType[Literal[128], Literal[Signedness.SIGNED]]
si255: TypeAlias = IntegerType[Literal[255], Literal[Signedness.SIGNED]]

# Type aliases for unsigned integers.
ui1: TypeAlias = IntegerType[Literal[1], Literal[Signedness.UNSIGNED]]
ui8: TypeAlias = IntegerType[Literal[8], Literal[Signedness.UNSIGNED]]
ui16: TypeAlias = IntegerType[Literal[16], Literal[Signedness.UNSIGNED]]
ui32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.UNSIGNED]]
ui64: TypeAlias = IntegerType[Literal[64], Literal[Signedness.UNSIGNED]]
ui128: TypeAlias = IntegerType[Literal[128], Literal[Signedness.UNSIGNED]]
ui255: TypeAlias = IntegerType[Literal[255], Literal[Signedness.UNSIGNED]]


class Index(FrontendType):
    """
    Represents an index type in the frontend. Should not be used explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.IndexType


class Float16(FrontendType):
    """
    Represents a 16-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float16Type

    def __add__(self, other: f16) -> f16:
        from xdsl.frontend.default.arith import addf

        return addf(self, other)

    def __sub__(self, other: f16) -> f16:
        from xdsl.frontend.default.arith import subf

        return subf(self, other)

    def __mul__(self, other: f16) -> f16:
        from xdsl.frontend.default.arith import mulf

        return mulf(self, other)


class Float32(FrontendType):
    """
    Represents a 32-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float32Type

    def __add__(self, other: f32) -> f32:
        from xdsl.frontend.default.arith import addf

        return addf(self, other)

    def __sub__(self, other: f32) -> f32:
        from xdsl.frontend.default.arith import subf

        return subf(self, other)

    def __mul__(self, other: f32) -> f32:
        from xdsl.frontend.default.arith import mulf

        return mulf(self, other)


class Float64(FrontendType):
    """
    Represents a 64-bit floating-point type in the frontend. Should not be used
    explicitly.
    """

    @staticmethod
    def to_xdsl() -> Callable[..., Any]:
        return builtin.Float64Type

    def __add__(self, other: f64) -> f64:
        from xdsl.frontend.default.arith import addf

        return addf(self, other)

    def __sub__(self, other: f64) -> f64:
        from xdsl.frontend.default.arith import subf

        return subf(self, other)

    def __mul__(self, other: f64) -> f64:
        from xdsl.frontend.default.arith import mulf

        return mulf(self, other)


index: TypeAlias = Index

# Type alias for floating-point types.
f16: TypeAlias = Float16
f32: TypeAlias = Float32
f64: TypeAlias = Float64

AnyFloat = TypeVar("AnyFloat", bound=Union[f16, f32, f64])
