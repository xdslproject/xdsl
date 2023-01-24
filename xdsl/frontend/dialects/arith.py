import xdsl.dialects.arith as arith

from typing import Callable, TypeVar, Union
from xdsl.frontend.dialects.builtin import index, i1, i32, i64, f16, f32, f64
from xdsl.ir import Operation

_Int = TypeVar("_Int", bound=Union[index, i1, i32, i64], covariant=True)


def addi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_addi() -> Callable[..., Operation]:
    return arith.Addi.get


def andi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_andi() -> Callable[..., Operation]:
    return arith.AndI.get


def cmpi(lhs: _Int, rhs: _Int, mnemonic: str) -> i1:
    pass


def resolve_cmpi() -> Callable[..., Operation]:
    return arith.Cmpi.from_mnemonic


def muli(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_muli() -> Callable[..., Operation]:
    return arith.Muli.get


def shli(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_shli() -> Callable[..., Operation]:
    return arith.ShLI.get


def shrsi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_shrsi() -> Callable[..., Operation]:
    return arith.ShRSI.get


def subi(lhs: _Int, rhs: _Int) -> _Int:
    pass


def resolve_subi() -> Callable[..., Operation]:
    return arith.Subi.get


_Float = TypeVar("_Float", bound=Union[f16, f32, f64], covariant=True)


def addf(lhs: _Float, rhs: _Float) -> _Float:
    pass


def resolve_addf() -> Callable[..., Operation]:
    return arith.Addf.get


def mulf(lhs: _Float, rhs: _Float) -> _Float:
    pass


def resolve_mulf() -> Callable[..., Operation]:
    return arith.Mulf.get


def subf(lhs: _Float, rhs: _Float) -> _Float:
    pass


def resolve_subf() -> Callable[..., Operation]:
    return arith.Subf.get
