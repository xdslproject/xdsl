from typing import IO

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Dialect

from .attrs import ObjectType
from .ops import (
    PyBinOp,
    PyConstOp,
    PyFunctionOp,
    PyReturnOp,
)
from .print import PythonPrinter


def print_python_source(module: ModuleOp, output: IO[str]) -> None:
    printer = PythonPrinter(stream=output)
    printer.print_module(module)


Python = Dialect(
    "python",
    [
        PyBinOp,
        PyConstOp,
        PyFunctionOp,
        PyReturnOp,
    ],
    [
        ObjectType,
    ],
)
"""
The Python AST dialect.
"""
