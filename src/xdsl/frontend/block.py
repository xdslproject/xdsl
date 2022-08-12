from typing import Callable, TypeVar, ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

def block(fn: Callable[P, T]) -> Callable[P, T]:
    return fn