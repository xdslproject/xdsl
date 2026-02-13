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
The Python semantics dialect.
"""
