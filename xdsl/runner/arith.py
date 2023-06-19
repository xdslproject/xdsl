from traitlets import Any
from xdsl.dialects.arith import Addi, Constant, Cmpi, Muli, Subi
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(Constant)
    def run_constant(
        self, interpreter: Interpreter, op: Constant, args: tuple[Any, ...]
    ):
        assert interpreter.interpreter_assert_isa(
            op.value,
            AnyIntegerAttr,
            f"arith.constant not implemented for {type(op.value)}",
        )
        return (op.value.value.data,)

    @impl(Subi)
    def run_subi(self, interpreter: Interpreter, op: Subi, args: tuple[Any, ...]):
        assert interpreter.interpreter_assert_isa(
            args,
            tuple[int, int],
            "arith.subi unexpected operand values.",
        )
        return (args[0] - args[1],)

    @impl(Addi)
    def run_addi(self, interpreter: Interpreter, op: Addi, args: tuple[Any, ...]):
        assert interpreter.interpreter_assert_isa(args, tuple[int, int])
        return (args[0] + args[1],)

    @impl(Muli)
    def run_muli(self, interpreter: Interpreter, op: Muli, args: tuple[Any, ...]):
        assert interpreter.interpreter_assert_isa(args, tuple[int, int])
        return (args[0] * args[1],)

    @impl(Cmpi)
    def run_cmpi(self, interpreter: Interpreter, op: Cmpi, args: tuple[Any, ...]):
        assert interpreter.interpreter_assert_isa(args, tuple[int, int])
        match op.predicate.value.data:
            case 0:  # "eq"
                return (args[0] == args[1],)
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )
