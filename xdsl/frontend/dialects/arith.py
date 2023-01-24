import xdsl.dialects.arith as arith

from typing import TypeVar, Union
from xdsl.frontend.dialects.builtin import index, i1, i32, i64, f16, f32, f64
from xdsl.ir import Operation

_Int = TypeVar("_Int", bound=Union[index, i1, i32, i64], covariant=True)


def addi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_addi() -> Operation:
    return arith.Addi.get


def andi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_andi() -> Operation:
    return arith.AndI.get

def cmpi(lhs: _Int, rhs: _Int, mnemonic: str) -> i1:
    pass


def resolve_cmpi() -> Operation:
    return arith.Cmpi.from_mnemonic


def muli(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_muli() -> Operation:
    return arith.Muli.get


def shli(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_shli() -> Operation:
    return arith.ShLI.get


def shrsi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_shrsi() -> Operation:
    return arith.ShRSI.get


def subi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_subi() -> Operation:
    return arith.Subi.get


_Float = TypeVar("_Float", bound=Union[f16, f32, f64], covariant=True)


def addf(lhs: _Float, rhs: _Float) -> _Float:
    pass


def resolve_addf() -> Operation:
    return arith.Addf.get


def mulf(lhs: _Float, rhs: _Float) -> _Float:
    pass


def resolve_mulf() -> Operation:
    return arith.Mulf.get


def subf(lhs: _Float, rhs: _Float) -> _Float:
    pass


def resolve_subf() -> Operation:
    return arith.Subf.get
