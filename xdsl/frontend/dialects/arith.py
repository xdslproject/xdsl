import xdsl.dialects.arith as arith

from typing import TypeVar, Union
from xdsl.frontend.dialects.builtin import IndexType, i1, i32, i64, f16, f32, f64
from xdsl.ir import Operation


_IntType = TypeVar("_IntType", bound=Union[IndexType, i1, i32, i64], covariant=True)


def addi(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_addi() -> Operation:
    return arith.Addi.get


def subi(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_subi() -> Operation:
    return arith.Subi.get


def muli(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_muli() -> Operation:
    return arith.Muli.get


def andi(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_andi() -> Operation:
    return arith.AndI.get


def shrsi(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_shrsi() -> Operation:
    return arith.ShRSI.get


def shli(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_shli() -> Operation:
    return arith.ShLI.get


def cmpi(lhs: _IntType, rhs: _IntType, mnemonic: str) -> i1:
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
