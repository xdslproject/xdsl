from collections.abc import Callable
from typing import TypeVar

import xdsl.dialects.arith as arith
from xdsl.frontend.dialects.builtin import f16, f32, f64, i1, i32, i64, index
from xdsl.ir import Operation

_Int = TypeVar("_Int", bound=index | i1 | i32 | i64)


def addi(lhs: _Int, rhs: _Int) -> _Int: ...


def resolve_addi() -> Callable[..., Operation]:
    return arith.AddiOp


def andi(lhs: _Int, rhs: _Int) -> _Int: ...


def resolve_andi() -> Callable[..., Operation]:
    return arith.AndIOp


def cmpi(lhs: _Int, rhs: _Int, mnemonic: str) -> i1: ...


def resolve_cmpi() -> Callable[..., Operation]:
    return arith.CmpiOp


def muli(lhs: _Int, rhs: _Int) -> _Int: ...


def resolve_muli() -> Callable[..., Operation]:
    return arith.MuliOp


def shli(lhs: _Int, rhs: _Int) -> _Int: ...


def resolve_shli() -> Callable[..., Operation]:
    return arith.ShLIOp


def shrsi(lhs: _Int, rhs: _Int) -> _Int: ...


def resolve_shrsi() -> Callable[..., Operation]:
    return arith.ShRSIOp


def subi(lhs: _Int, rhs: _Int) -> _Int: ...


def resolve_subi() -> Callable[..., Operation]:
    return arith.SubiOp


_Float = TypeVar("_Float", bound=f16 | f32 | f64)


def addf(lhs: _Float, rhs: _Float) -> _Float: ...


def resolve_addf() -> Callable[..., Operation]:
    return arith.AddfOp


def mulf(lhs: _Float, rhs: _Float) -> _Float: ...


def resolve_mulf() -> Callable[..., Operation]:
    return arith.MulfOp


def subf(lhs: _Float, rhs: _Float) -> _Float: ...


def resolve_subf() -> Callable[..., Operation]:
    return arith.SubfOp
