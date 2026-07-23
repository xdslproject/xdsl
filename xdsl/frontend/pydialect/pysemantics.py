from ast import Add, Div, FloorDiv, LShift, MatMult, Mod, Mult, Pow, RShift, Sub
from typing import Any

OPERATOR_TO_DUNDER = {
    # And   : "__and__",
    # Or   : "__or__",
    Add: "__add__",
    Sub: "__sub__",
    Mult: "__mul__",
    MatMult: "__matmul__",
    Div: "__truediv__",
    Mod: "__mod__",
    Pow: "__pow__",
    LShift: "__lshift__",
    RShift: "__rshift__",
    FloorDiv: "__floordiv__",
    # BitOr    : "__and__",
    # BitXor   : "__xor__",
    # BitAnd   : "__and__",
    # Invert   : "__invert__",
    # Not      : "__add__",
    # UAdd     : "__add__",
    # USub     : "__add__",
    # Eq     : "__eq__",
    # NotEq  : "__ne__",
    # Lt   : "__lt__",
    # LtE   : "__le__",
    # Gt   : "__gt__",
    # GtE   : "__ge__",
    # Is   : "____",
    # IsNot   : "__add__",
    # In   : "__contains__",
    # NotIn   : "__add__"
}

TYPES_TO_NAME = {int: "int", float: "float"}


def dunder_op_name(v: Any):
    return OPERATOR_TO_DUNDER[type(v)]


def type_name(v: Any) -> str:
    return TYPES_TO_NAME[type(v)]
