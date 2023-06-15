from traitlets import Any
from xdsl.dialects.print import PrintLnOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class PrintFunctions(InterpreterFunctions):
    @impl(PrintLnOp)
    def run_module(
        self, interpreter: Interpreter, op: PrintLnOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        format_str = op.format_str.data
        print(format_str.format(*args))
        return ()
