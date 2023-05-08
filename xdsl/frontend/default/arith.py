import xdsl.dialects.arith as arith

from typing import Callable
from xdsl.frontend.frontend import frontend_op
from xdsl.frontend.default.builtin import IntegerType, i1, AnyFloat
from xdsl.frontend.default.frontend import defaultFrontend
from xdsl.frontend.exception import FrontendProgramException
from xdsl.ir import Operation


@frontend_op(defaultFrontend, arith.Addi)
def addi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs + rhs


def resolve_addi() -> Callable[..., Operation]:
    return arith.Addi


@frontend_op(defaultFrontend, arith.AndI)
def andi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lsh & rhs


def resolve_andi() -> Callable[..., Operation]:
    return arith.AndI.get


@frontend_op(defaultFrontend, arith.Cmpi)
def cmpi(lhs: IntegerType, rhs: IntegerType, mnemonic: str) -> i1:
    match mnemonic:
        case "eq":
            return lhs == rhs
        case "ne":
            return lhs != rhs
        case "slt":
            return lhs < rhs
        case "sle":
            return lhs <= rhs
        case "sgt":
            return lhs > rhs
        case "sge":
            return lhs >= rhs
        case "ult":
            return lhs < rhs
        case "ule":
            return lhs <= rhs
        case "ugt":
            return lhs > rhs
        case "uge":
            return lhs >= rhs
        case _:
            raise FrontendProgramException(f"Unknown predicate {mnemonic}")


def resolve_cmpi() -> Callable[..., Operation]:
    return arith.Cmpi.get


@frontend_op(defaultFrontend, arith.Muli)
def muli(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs * rhs


def resolve_muli() -> Callable[..., Operation]:
    return arith.Muli.get


@frontend_op(defaultFrontend, arith.ShLI)
def shli(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs << rhs


def resolve_shli() -> Callable[..., Operation]:
    return arith.ShLI.get


@frontend_op(defaultFrontend, arith.ShRSI)
def shrsi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs >> rhs


def resolve_shrsi() -> Callable[..., Operation]:
    return arith.ShRSI.get


@frontend_op(defaultFrontend, arith.Subi)
def subi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs - rhs


def resolve_subi() -> Callable[..., Operation]:
    return arith.Subi.get


@frontend_op(defaultFrontend, arith.Addf)
def addf(lhs: AnyFloat, rhs: AnyFloat) -> AnyFloat:
    return lhs + rhs


def resolve_addf() -> Callable[..., Operation]:
    return arith.Addf.get


@frontend_op(defaultFrontend, arith.Mulf)
def mulf(lhs: AnyFloat, rhs: AnyFloat) -> AnyFloat:
    return lhs * rhs


def resolve_mulf() -> Callable[..., Operation]:
    return arith.Mulf.get


@frontend_op(defaultFrontend, arith.Subf)
def subf(lhs: AnyFloat, rhs: AnyFloat) -> AnyFloat:
    return lhs - rhs


def resolve_subf() -> Callable[..., Operation]:
    return arith.Subf.get
