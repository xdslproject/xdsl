from traitlets import Any
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import IntegerAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(Constant)
    def run_module(
        self, interpreter: Interpreter, op: Constant, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if isinstance(op.value, IntegerAttr):
            return (op.value.value.data,)  # type: ignore
