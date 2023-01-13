import ast
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")


class Const(Generic[_T]):
    """
    Marks the value as a compile-time constant and can be used in any frontend.
    Compile-time constants can be expressions and are evaluated at compile time.
    They can only live in the global scope of the program.

    ```
    a: Const[i32] = 123             # i32 constant equal to 123
    b: Const[i64] = len([1, 2, 3])  # i64 constant equal to 3

    c: Const[i64] = 123 / 0         # compile-time error, division by zero

    def foo():
        d: Const[i32] = 0           # compile-time error, constants inside
                                    # functions are not (yet) supported.

        a = 34                      # compile-time error, cannot assign to constants.

        b: i64 = 23                 # here, 'b' is shadowed, and therefore the
        b = 45                      # assignement succeeds.
    ```
    """

    @staticmethod
    def check(node: ast.expr) -> bool:
        """Returns `True` if the AST node is a Const type."""
        return isinstance(node, ast.Subscript) and isinstance(
            node.value, ast.Name) and node.value.id == Const.__name__
