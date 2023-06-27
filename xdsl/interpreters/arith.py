from typing import cast
from xdsl.dialects.arith import Addi, Constant, Cmpi, Muli, Subi
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(Constant)
    def run_constant(self, interpreter: Interpreter, op: Constant, args: tuple[()]):
        value = op.value
        interpreter.interpreter_assert(
            isa(op.value, AnyIntegerAttr),
            f"arith.constant not implemented for {type(op.value)}",
        )
        value = cast(AnyIntegerAttr, op.value)
        return (value.value.data,)

    @impl(Subi)
    def run_subi(self, interpreter: Interpreter, op: Subi, args: tuple[int, int]):
        return (args[0] - args[1],)

    @impl(Addi)
    def run_addi(self, interpreter: Interpreter, op: Addi, args: tuple[int, int]):
        return (args[0] + args[1],)

    @impl(Muli)
    def run_muli(self, interpreter: Interpreter, op: Muli, args: tuple[int, int]):
        return (args[0] * args[1],)

    @impl(Cmpi)
    def run_cmpi(self, interpreter: Interpreter, op: Cmpi, args: tuple[int, int]):
        match op.predicate.value.data:
            case 0:  # "eq"
                return (args[0] == args[1],)
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )
