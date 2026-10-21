from typing import Any

from xdsl.dialects.builtin import FloatAttr
from xdsl.dialects.printf import PrintFormatOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.utils.hints import isa


@register_impls
class PrintfFunctions(InterpreterFunctions):
    @impl(PrintFormatOp)
    def run_println(
        self, interpreter: Interpreter, op: PrintFormatOp, args: tuple[Any, ...]
    ):
        args = tuple(arg.value.data if isa(arg, FloatAttr) else arg for arg in args)
        print(op.format_str.data.format(*args), file=interpreter.file, end="")
        return ()
