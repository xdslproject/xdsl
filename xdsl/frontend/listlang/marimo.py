from io import StringIO

import marimo as mo

from xdsl.dialects import builtin
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.printf import PrintfFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.interpreters.tensor import TensorFunctions


def interp(module: builtin.ModuleOp) -> str:
    _io = StringIO()

    _i = Interpreter(module=module, file=_io)
    _i.register_implementations(ArithFunctions())
    _i.register_implementations(ScfFunctions())
    _i.register_implementations(PrintfFunctions())
    _i.register_implementations(TensorFunctions())
    _i.run_ssacfg_region(module.body, ())

    # Lowercase to avoid capitals in `True` and `False` printing of bools.
    # Safe since we never print strings other than `True` and `False`.
    return _io.getvalue().lower()


def rust_md(code: str) -> mo.Html:
    return mo.md("`" * 3 + "rust\n" + code + "\n" + "`" * 3)
