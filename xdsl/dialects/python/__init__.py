from xdsl.ir import Dialect

from .attrs import ObjectType
from .ops import (
    BinaryOpOp,
    LoadConstOp,
    PythonFunctionOp,
    ReturnValueOp,
)

Python = Dialect(
    "python",
    [
        PythonFunctionOp,
        LoadConstOp,
        BinaryOpOp,
        ReturnValueOp,
    ],
    [
        ObjectType,
    ],
)
"""
The Python Bytecode dialect.
"""
