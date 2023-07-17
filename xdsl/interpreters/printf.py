from typing import Any

from xdsl.dialects.printf import PrintFormatOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls


@register_impls
class PrintfFunctions(InterpreterFunctions):
    @impl(PrintFormatOp)
    def run_println(
        self, interpreter: Interpreter, op: PrintFormatOp, args: tuple[Any, ...]
    ):
        print(op.format_str.data.format(*args), file=interpreter.file)
        return ()
