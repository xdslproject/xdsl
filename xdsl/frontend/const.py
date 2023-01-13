import ast
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")


class Const(Generic[_T]):
    """
    Marks the value as a compile-time constant and can be used in any frontend.
    Compile-time constants can be expressions and are evaluated at compile time.
    They can live in the global scope of the program.
    """

    @staticmethod
    def check(node: ast.expr) -> bool:
        """Returns `True` if the AST node is a Const type."""
        return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == Const.__name__
