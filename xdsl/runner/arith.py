from traitlets import Any
from xdsl.dialects.arith import Addi, Constant, Cmpi, Muli, Subi
from xdsl.dialects.builtin import IntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(Constant)
    def run_constant(
        self, interpreter: Interpreter, op: Constant, args: tuple[Any, ...]
    ):
        if isinstance(op.value, IntegerAttr):
            return (op.value.value.data,)
        raise InterpretationError(
            f"arith.constant not implemented for {type(op.value)}"
        )

    @impl(Subi)
    def run_subi(self, interpreter: Interpreter, op: Subi, args: tuple[Any, ...]):
        if not isa(args, tuple[int | float, int | float]):
            raise InterpretationError(f"arith.subi unexpected operand values.")
        return (args[0] - args[1],)

    @impl(Addi)
    def run_addi(self, interpreter: Interpreter, op: Addi, args: tuple[Any, ...]):
        if not isa(args, tuple[int | float, int | float]):
            raise InterpretationError()
        return (args[0] + args[1],)

    @impl(Muli)
    def run_muli(self, interpreter: Interpreter, op: Muli, args: tuple[Any, ...]):
        if not isa(args, tuple[int | float, int | float]):
            raise InterpretationError()
        return (args[0] * args[1],)

    @impl(Cmpi)
    def run_cmpi(self, interpreter: Interpreter, op: Cmpi, args: tuple[Any, ...]):
        if not isa(args, tuple[int | float, int | float]):
            raise InterpretationError()
        match op.predicate.value.data:
            # "eq"
            case 0:
                return (args[0] == args[1],)  # type: ignore
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )
