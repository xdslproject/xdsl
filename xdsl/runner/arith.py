from traitlets import Any
from xdsl.dialects.arith import Constant, Cmpi
from xdsl.dialects.builtin import IntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.exceptions import InterpretationError


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(Constant)
    def run_constant(
        self, interpreter: Interpreter, op: Constant, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if isinstance(op.value, IntegerAttr):
            return (op.value.value.data,)  # type: ignore

    @impl(Cmpi)
    def run_cmpi(
        self, interpreter: Interpreter, op: Cmpi, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        match op.predicate.value.data:
            # "eq"
            case 0:
                return (args[0] == args[1],)  # type: ignore
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )
