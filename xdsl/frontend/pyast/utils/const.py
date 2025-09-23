import ast
from typing import Generic

from typing_extensions import TypeVar

_T = TypeVar("_T")


class Const(Generic[_T]):
    """
    Marks the value as a compile-time constant and can be used for any type
    defined by the frontend. Compile-time constants can be expressions and are
    evaluated at compile time. They cannot be redefined in the same program or
    shadowed by other variables.

    ```
    a: Const[i64] = 123             # i64 constant equal to 123
    b: Const[i32] = len([1, 2, 3])  # i32 constant equal to 3

    c: Const[f32] = 123 / 0         # compile-time error, division by zero

    d: Const[i64] = a + b           # i64 constant, equal to 6

    def foo(a: i64):                # compile-time error, cannot reuse the name
                                    # of a constant

        d = 3                       # compile-time error, cannot assign to a constant
        e: Const[i16] = b + 2       # i16 constant, equal to 5
    ```
    """

    pass


def is_constant(node: ast.expr) -> bool:
    """Returns `True` if the AST node is a Const type."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == Const.__name__
    )


def is_constant_stmt(node: ast.stmt) -> bool:
    """Returns `True` if the AST statement is a Const expression."""
    return isinstance(node, ast.AnnAssign) and is_constant(node.annotation)
