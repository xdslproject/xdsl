import ast
from typing import Any


def block(*params: Any):
    """
    Decorator used to mark function as a basic block.
    ```
    def foo(a: int) -> int:
        @block
        def bb0(x: int):
            y: int = x + 2
            bb1(y)

        @block
        def bb1(z: int):
            return z

        # Entry-point.
        bb0(a)
    ```
    """

    def decorate(*params: Any):
        return None

    return decorate


def is_block(node: ast.FunctionDef) -> bool:
    return (
        len(node.decorator_list) == 1
        and isinstance(name := node.decorator_list[0], ast.Name)
        and name.id == "block"
    )
