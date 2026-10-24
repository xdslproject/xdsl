"""
The py dialect is intended to represent a Python-like language.

It is supposed to be a middle-ground between Python and bytecode
"""

from xdsl.ir import Dialect

from .attrs import (
    ConstantValue,
    ObjectType,
)
from .ops import (
    CallOp,
    CastOp,
    ConstantOp,
    FuncOp,
    PassOp,
    ReturnOp,
)

Py = Dialect(
    "py",
    [
        CallOp,
        CastOp,
        ConstantOp,
        FuncOp,
        PassOp,
        ReturnOp,
    ],
    [
        ConstantValue,
        ObjectType,
    ],
)
