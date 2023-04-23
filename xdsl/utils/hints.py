from inspect import isclass
from types import UnionType
from typing import Annotated, Any, TypeGuard, TypeVar, Union, cast, get_args, get_origin
from xdsl.ir import ParametrizedAttribute

from xdsl.utils.exceptions import VerifyException

_T = TypeVar("_T")


def isa(arg: Any, hint: type[_T]) -> TypeGuard[_T]:
    """
    Check if `arg` is of the type described by `hint`.
    For now, only lists, dictionaries, unions,
    and non-generic classes are supported for type hints.
    """
    if hint is Any:
        return True

    origin = get_origin(hint)

    # get_origin checks that hint is not a parametrized generic
    if isclass(hint) and (origin is None):
        return isinstance(arg, hint)

    if origin is list:
        if not isinstance(arg, list):
            return False
        arg_list: list[Any] = cast(list[Any], arg)
        (elem_hint,) = get_args(hint)
        return all(isa(elem, elem_hint) for elem in arg_list)

    if origin is tuple:
        if not isinstance(arg, tuple):
            return False
        elem_hints = get_args(hint)
        arg_tuple: tuple[Any, ...] = cast(tuple[Any, ...], arg)
        if len(elem_hints) == 2 and elem_hints[1] is ...:
            return all(isa(elem, elem_hints[0]) for elem in arg_tuple)
        else:
            return len(elem_hints) == len(arg_tuple) and all(
                isa(elem, hint) for elem, hint in zip(arg_tuple, elem_hints)
            )

    if origin is dict:
        if not isinstance(arg, dict):
            return False
        arg_dict: dict[Any, Any] = cast(dict[Any, Any], arg)
        key_hint, value_hint = get_args(hint)
        return all(
            isa(key, key_hint) and isa(value, value_hint)
            for key, value in arg_dict.items()
        )

    if origin in [Union, UnionType]:
        return any(isa(arg, union_arg) for union_arg in get_args(hint))

    from xdsl.irdl import GenericData, irdl_to_attr_constraint

    if (origin is not None) and issubclass(origin, GenericData | ParametrizedAttribute):
        constraint = irdl_to_attr_constraint(hint)
        try:
            constraint.verify(arg)
            return True
        except VerifyException:
            return False

    raise ValueError(f"isa: unsupported type hint '{hint}' {get_origin(hint)}")


def assert_isa(arg: Any, hint: type[_T]) -> TypeGuard[_T]:
    """
    Check if `arg` is of the type described by `hint`.
    For now, only lists, dictionaries, unions,
    and non-generic classes are supported for type hints.
    """

    if not isa(arg, hint):
        raise ValueError(
            f"Expected value of type {hint}, got value of type {type(arg).__name__}"
        )
    return True


annotated_type = type(Annotated[int, 0])
"""This is the type of an Annotated object."""


class _Class:
    @property
    def property(self):
        pass


PropertyType = type(_Class.property)
"""The type of a property method."""
