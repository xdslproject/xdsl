import ast
from typing import Any

OPERATOR_TO_DUNDER = {
    # And   : "__and__",
    # Or   : "__or__",
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.FloorDiv: "__floordiv__",
    # BitOr    : "__and__",
    # BitXor   : "__xor__",
    # BitAnd   : "__and__",
    # Invert   : "__invert__",
    # Not      : "__add__",
    # UAdd     : "__add__",
    # USub     : "__add__",
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
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
