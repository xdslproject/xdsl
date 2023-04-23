import xdsl.dialects.arith as arith

from typing import Callable, TypeVar, Union
from xdsl.frontend.dialects.builtin import index, i1, i32, i64, f16, f32, f64
from xdsl.ir import Operation

_Int = TypeVar("_Int", bound=Union[index, i1, i32, i64])


def addi(lhs: _Int, rhs: _Int) -> _Int:
    ...


def resolve_addi() -> Callable[..., Operation]:
    return arith.Addi


def andi(lhs: _Int, rhs: _Int) -> _Int:
    ...


def resolve_andi() -> Callable[..., Operation]:
    return arith.AndI.get


def cmpi(lhs: _Int, rhs: _Int, mnemonic: str) -> i1:
    ...


def resolve_cmpi() -> Callable[..., Operation]:
    return arith.Cmpi.get


def muli(lhs: _Int, rhs: _Int) -> _Int:
    ...


def resolve_muli() -> Callable[..., Operation]:
    return arith.Muli.get


def shli(lhs: _Int, rhs: _Int) -> _Int:
    ...


def resolve_shli() -> Callable[..., Operation]:
    return arith.ShLI.get


def shrsi(lhs: _Int, rhs: _Int) -> _Int:
    ...


def resolve_shrsi() -> Callable[..., Operation]:
    return arith.ShRSI.get


def subi(lhs: _Int, rhs: _Int) -> _Int:
    ...


def resolve_subi() -> Callable[..., Operation]:
    return arith.Subi.get


_Float = TypeVar("_Float", bound=Union[f16, f32, f64])


def addf(lhs: _Float, rhs: _Float) -> _Float:
    ...


def resolve_addf() -> Callable[..., Operation]:
    return arith.Addf.get


def mulf(lhs: _Float, rhs: _Float) -> _Float:
    ...


def resolve_mulf() -> Callable[..., Operation]:
    return arith.Mulf.get


def subf(lhs: _Float, rhs: _Float) -> _Float:
    ...


def resolve_subf() -> Callable[..., Operation]:
    return arith.Subf.get
