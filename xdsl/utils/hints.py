from inspect import isclass
from types import UnionType
from typing import (Annotated, Any, TypeGuard, TypeVar, Union, cast, get_args,
                    get_origin)

_T = TypeVar("_T")


def is_satisfying_hint(arg: Any, hint: type[_T]) -> TypeGuard[_T]:
    """
    Check if `arg` is of the type described by `hint`.
    For now, only lists, tuples, sets, dictionaries, unions,
    and non-generic classes are supported for type hints.
    """
    if hint is Any:
        return True

    # get_origin checks that hint is not a parametrized generic
    if isclass(hint) and (get_origin(hint) is None):
        return isinstance(arg, hint)

    if get_origin(hint) is list:
        if not isinstance(arg, list):
            return False
        arg_list: list[Any] = cast(list[Any], arg)
        elem_hint, = get_args(hint)
        return all(is_satisfying_hint(elem, elem_hint) for elem in arg_list)

    if get_origin(hint) is dict:
        if not isinstance(arg, dict):
            return False
        arg_dict: dict[Any, Any] = cast(dict[Any, Any], arg)
        key_hint, value_hint = get_args(hint)
        return all(
            is_satisfying_hint(key, key_hint)
            and is_satisfying_hint(value, value_hint)
            for key, value in arg_dict.items())

    if get_origin(hint) in [Union, UnionType]:
        return any(
            is_satisfying_hint(arg, union_arg) for union_arg in get_args(hint))

    raise ValueError(f"is_satisfying_hint: unsupported type hint '{hint}'")


annotated_type = type(Annotated[int, 0])
"""This is the type of an Annotated object."""
