from io import StringIO
from typing import Any

import marimo as mo

from xdsl.dialects import builtin
from xdsl.dialects.printf import PrintFormatOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.utils.hints import isa


@register_impls
class PrintfFunctions(InterpreterFunctions):
    def _format_arg(self, fmt_val: Any, arg: Any) -> str:
        if isa(fmt_val.type, builtin.I1):
            return "true" if arg else "false"
        return str(arg)

    @impl(PrintFormatOp)
    def run_println(
        self, interpreter: Interpreter, op: PrintFormatOp, args: tuple[Any, ...]
    ):
        pretty_args = tuple(
            self._format_arg(fmt_val, arg)
            for fmt_val, arg in zip(op.format_vals, args, strict=True)
        )

        print(
            op.format_str.data.format(*pretty_args),
            file=interpreter.file,
            end="",
        )
        return ()


def interp(module: builtin.ModuleOp) -> str:
    _io = StringIO()

    _i = Interpreter(module=module, file=_io)
    _i.register_implementations(ArithFunctions())
    _i.register_implementations(ScfFunctions())
    _i.register_implementations(PrintfFunctions())
    _i.register_implementations(TensorFunctions())
    _i.run_ssacfg_region(module.body, ())

    return _io.getvalue()


def rust_md(code: str) -> mo.Html:
    return mo.md("`" * 3 + "rust\n" + code + "\n" + "`" * 3)
