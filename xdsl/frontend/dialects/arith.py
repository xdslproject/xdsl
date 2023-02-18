import xdsl.dialects.arith as arith

from typing import Callable, TypeVar, Union
from xdsl.frontend.dialects.builtin import index, i1, i32, i64, f16, f32, f64
from xdsl.ir import Operation

Int = TypeVar("Int", bound=Union[index, i1, i32, i64])


def addi(lhs: Int, rhs: Int) -> Int:
    ...


def resolve_addi() -> Callable[..., Operation]:
    return arith.Addi.get


def andi(lhs: Int, rhs: Int) -> Int:
    ...


def resolve_andi() -> Callable[..., Operation]:
    return arith.AndI.get


def cmpi(lhs: Int, rhs: Int, mnemonic: str) -> i1:
    ...


def resolve_cmpi() -> Callable[..., Operation]:
    return arith.Cmpi.from_mnemonic


def muli(lhs: Int, rhs: Int) -> Int:
    ...


def resolve_muli() -> Callable[..., Operation]:
    return arith.Muli.get


def shli(lhs: Int, rhs: Int) -> Int:
    ...


def resolve_shli() -> Callable[..., Operation]:
    return arith.ShLI.get


def shrsi(lhs: Int, rhs: Int) -> Int:
    ...


def resolve_shrsi() -> Callable[..., Operation]:
    return arith.ShRSI.get


def subi(lhs: Int, rhs: Int) -> Int:
    ...


def resolve_subi() -> Callable[..., Operation]:
    return arith.Subi.get


Float = TypeVar("Float", bound=Union[f16, f32, f64])


def addf(lhs: Float, rhs: Float) -> Float:
    ...


def resolve_addf() -> Callable[..., Operation]:
    return arith.Addf.get


def mulf(lhs: Float, rhs: Float) -> Float:
    ...


def resolve_mulf() -> Callable[..., Operation]:
    return arith.Mulf.get


def subf(lhs: Float, rhs: Float) -> Float:
    ...


def resolve_subf() -> Callable[..., Operation]:
    return arith.Subf.get
