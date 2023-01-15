import ast
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")


class Const(Generic[_T]):
    """
    Marks the value as a compile-time constant and can be used in any frontend.
    Compile-time constants can be expressions and are evaluated at compile time.
    They cannot be redefined in the same program but can be shadowed by .

    ```
    a: Const[i64] = 123             # i64 constant equal to 123
    b: Const[i64] = len([1, 2, 3])  # i64 constant equal to 3

    c: Const[i64] = 123 / 0         # compile-time error, division by zero

    d: Const[i64] = a + b           # i64 constant, equal to 6

    def foo(a: i64):                # compile-time error, cannot reuse the name
                                    # of the constant

        d = 3                       # compile-time error, cannot assign to constant
        e: Const[i32] = b + 2       # i32 constant, equal to 5
    ```
    """
    pass


def is_constant(node: ast.expr) -> bool:
    """Returns `True` if the AST node is a Const type."""
    return isinstance(node, ast.Subscript) and isinstance(
        node.value, ast.Name) and node.value.id == Const.__name__
