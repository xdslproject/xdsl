import xdsl.dialects.arith as arith

from typing import Callable
from xdsl.frontend.frontend import frontend_op
from xdsl.frontend.default.builtin import IntegerType, i1, AnyFloat
from xdsl.frontend.default.default_frontend import defaultFrontend
from xdsl.frontend.exception import FrontendProgramException
from xdsl.ir import Operation


@frontend_op(defaultFrontend, arith.Addi)
def addi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs + rhs


@frontend_op(defaultFrontend, arith.AndI)
def andi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lsh & rhs


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


@frontend_op(defaultFrontend, arith.Muli)
def muli(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs * rhs


@frontend_op(defaultFrontend, arith.ShLI)
def shli(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs << rhs


@frontend_op(defaultFrontend, arith.ShRSI)
def shrsi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs >> rhs


@frontend_op(defaultFrontend, arith.Subi)
def subi(lhs: IntegerType, rhs: IntegerType) -> IntegerType:
    return lhs - rhs


@frontend_op(defaultFrontend, arith.Addf)
def addf(lhs: AnyFloat, rhs: AnyFloat) -> AnyFloat:
    return lhs + rhs


@frontend_op(defaultFrontend, arith.Mulf)
def mulf(lhs: AnyFloat, rhs: AnyFloat) -> AnyFloat:
    return lhs * rhs


@frontend_op(defaultFrontend, arith.Subf)
def subf(lhs: AnyFloat, rhs: AnyFloat) -> AnyFloat:
    return lhs - rhs
