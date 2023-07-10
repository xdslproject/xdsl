from typing import cast

from xdsl.dialects import arith
from xdsl.dialects.builtin import AnyFloatAttr, AnyIntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(arith.Constant)
    def run_constant(
        self, interpreter: Interpreter, op: arith.Constant, args: tuple[()]
    ):
        value = op.value
        interpreter.interpreter_assert(
            isa(op.value, AnyIntegerAttr | AnyFloatAttr),
            f"arith.constant not implemented for {type(op.value)}",
        )
        value = cast(AnyIntegerAttr, op.value)
        return (value.value.data,)

    @impl(arith.Subi)
    def run_subi(self, interpreter: Interpreter, op: arith.Subi, args: tuple[int, int]):
        return (args[0] - args[1],)

    @impl(arith.Addi)
    def run_addi(self, interpreter: Interpreter, op: arith.Addi, args: tuple[int, int]):
        return (args[0] + args[1],)

    @impl(arith.Muli)
    def run_muli(self, interpreter: Interpreter, op: arith.Muli, args: tuple[int, int]):
        return (args[0] * args[1],)

    @impl(arith.Subf)
    def run_subf(
        self, interpreter: Interpreter, op: arith.Subf, args: tuple[float, float]
    ):
        return (args[0] - args[1],)

    @impl(arith.Addf)
    def run_addf(
        self, interpreter: Interpreter, op: arith.Addf, args: tuple[float, float]
    ):
        return (args[0] + args[1],)

    @impl(arith.Mulf)
    def run_mulf(
        self, interpreter: Interpreter, op: arith.Mulf, args: tuple[float, float]
    ):
        return (args[0] * args[1],)

    @impl(arith.Cmpi)
    def run_cmpi(self, interpreter: Interpreter, op: arith.Cmpi, args: tuple[int, int]):
        match op.predicate.value.data:
            case 0:  # "eq"
                return (args[0] == args[1],)
            case 1:  # "ne"
                return (args[0] != args[1],)
            case 2:  # "slt"
                return (args[0] < args[1],)
            case 3:  # "sle"
                return (args[0] <= args[1],)
            case 4:  # "sgt"
                return (args[0] > args[1],)
            case 5:  # "sge"
                return (args[0] >= args[1],)
            case 6:  # "ult"
                return (args[0] < args[1],)
            case 7:  # "ule"
                return (args[0] <= args[1],)
            case 8:  # "ugt"
                return (args[0] > args[1],)
            case 9:  # "uge"
                return (args[0] >= args[1],)
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )
