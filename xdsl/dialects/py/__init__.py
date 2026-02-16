from xdsl.ir import Dialect

from .ops import (
    PyBinOp,
    PyConstOp,
)

Py = Dialect(
    "py",
    [
        PyBinOp,
        PyConstOp,
    ],
    [],
)
"""
This module contains the definition of the Python semantics dialect.

We only guarantee preservation of most[1] Python semantics, but do not guarantee
preservation of AST or bytecode.

The Python dialect is under heavy development, and the definitions are not yet stable!

[1]: Assumptions:
        1. We assume no exceptions are raised in the code.
"""
