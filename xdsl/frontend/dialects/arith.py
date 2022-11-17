import xdsl.dialects.arith as arith

from typing import TypeVar, Union
from xdsl.dialects.builtin import Signedness
from xdsl.frontend.dialects.builtin import IntegerType, i1, f16, f32, f64
from xdsl.ir import Operation


# Type parameters for integers.
_Width = TypeVar("_Width", bound=int, covariant=True)
_Signedness = TypeVar("_Signedness", bound=Signedness, covariant=True)


def addi(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness]) -> IntegerType[_Width, _Signedness]:
    pass


def resolve_addi() -> Operation:
    return arith.Addi.get


def subi(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness]) -> IntegerType[_Width, _Signedness]:
    pass


def resolve_subi() -> Operation:
    return arith.Subi.get


def muli(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness]) -> IntegerType[_Width, _Signedness]:
    pass


def resolve_muli() -> Operation:
    return arith.Muli.get


def andi(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness]) -> IntegerType[_Width, _Signedness]:
    pass


def resolve_andi() -> Operation:
    return arith.AndI.get


def shrsi(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness]) -> IntegerType[_Width, _Signedness]:
    pass


def resolve_shrsi() -> Operation:
    return arith.ShRSI.get


def shli(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness]) -> IntegerType[_Width, _Signedness]:
    pass


def resolve_shli() -> Operation:
    return arith.ShLI.get


def cmpi(lhs: IntegerType[_Width, _Signedness], rhs: IntegerType[_Width, _Signedness], mnemonic: str) -> i1:
    pass


def resolve_cmpi() -> Operation:
    return arith.Cmpi.from_mnemonic


_FloatType = TypeVar("_FloatType", bound=Union[f16, f32, f64], covariant=True)


def addf(lhs: _FloatType, rhs: _FloatType) -> _FloatType:
    pass

def resolve_addf() -> Operation:
    return arith.Addf.get


def subf(lhs: _FloatType, rhs: _FloatType) -> _FloatType:
    pass


def resolve_subf() -> Operation:
    return arith.Subf.get


def mulf(lhs: _FloatType, rhs: _FloatType) -> _FloatType:
    pass


def resolve_mulf() -> Operation:
    return arith.Mulf.get
