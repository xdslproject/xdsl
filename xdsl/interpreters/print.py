from typing import Any

from xdsl.dialects.print import PrintLnOp

from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class PrintFunctions(InterpreterFunctions):
    @impl(PrintLnOp)
    def run_println(
        self, interpreter: Interpreter, op: PrintLnOp, args: tuple[Any, ...]
    ):
        print(op.format_str.data.format(*args), file=interpreter.file)
        return ()
