import inspect
from typing import (Annotated, get_args, get_origin, Union, Any, cast, TypeVar,
                    TypeGuard)

_T = TypeVar("_T")


def is_satisfying_hint(arg: Any, hint: type[_T]) -> TypeGuard[_T]:
    """
    Check if `arg` is of the type described by `hint`.
    For now, only lists, dictionaries, unions, and non-generic
    classes are supported for type hints.
    """
    if hint is Any:
        return True

    # get_origin checks that hint is not a parametrized generic
    if inspect.isclass(hint) and (get_origin(hint) is None):
        return isinstance(arg, hint)

    if get_origin(hint) == list:
        if not isinstance(arg, list):
            return False
        arg = cast(list[Any], arg)  # This is to make the type checker happy
        if len(arg) == 0:
            return True
        elem_hint = get_args(hint)[0]
        return all(is_satisfying_hint(elem, elem_hint) for elem in arg)

    if get_origin(hint) == dict:
        if not isinstance(arg, dict):
            return False
        arg = cast(list[Any], arg)  # This is to make the type checker happy
        if len(arg) == 0:
            return True
        key_hint = get_args(hint)[0]
        value_hint = get_args(hint)[1]
        return all(
            is_satisfying_hint(key, key_hint)
            and is_satisfying_hint(value, value_hint)
            for key, value in arg.items())

    if get_origin(hint) == Union:
        for union_arg in get_args(hint):
            if is_satisfying_hint(arg, union_arg):
                return True
        return False

    raise ValueError(f"is_satisfying_hint: unsupported type hint '{hint}'")


annotated_type = type(Annotated[int, 0])
"""This is the type of an Annotated object."""
