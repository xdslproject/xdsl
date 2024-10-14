import types
from collections.abc import Iterable, Sequence
from inspect import isclass
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from xdsl.ir import ParametrizedAttribute
from xdsl.utils.exceptions import VerifyException

_T = TypeVar("_T")


def isa(arg: Any, hint: type[_T]) -> TypeGuard[_T]:
    from xdsl.irdl import ConstraintContext

    """
    Check if `arg` is of the type described by `hint`.
    For now, only lists, dictionaries, unions,
    and non-generic classes are supported for type hints.
    """
    if hint is Any:  # pyright: ignore[reportUnnecessaryComparison]
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

    if origin is set:
        if not isinstance(arg, set):
            return False
        arg_set: set[Any] = cast(set[Any], arg)
        (elem_hint,) = get_args(hint)
        return all(isa(elem, elem_hint) for elem in arg_set)

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

    if origin in [Union, types.UnionType]:
        return any(isa(arg, union_arg) for union_arg in get_args(hint))

    if origin is Literal:
        return arg in get_args(hint)

    if (origin is not None) and issubclass(origin, Sequence):
        if not isinstance(arg, Sequence):
            return False
        arg_seq = cast(Sequence[Any], arg)
        (elem_hint,) = get_args(hint)
        return all(isa(elem, elem_hint) for elem in arg_seq)

    from xdsl.irdl import GenericData, irdl_to_attr_constraint

    if (origin is not None) and issubclass(origin, GenericData | ParametrizedAttribute):
        constraint = irdl_to_attr_constraint(hint)
        try:
            constraint.verify(arg, ConstraintContext())
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


def get_type_var_from_generic_class(cls: type[Any]) -> tuple[TypeVar, ...]:
    """Return the `TypeVar` used in the class `Generic` parent."""
    cls_orig_bases: Iterable[Any] = getattr(cls, "__orig_bases__")
    for orig_base in cls_orig_bases:
        if get_origin(orig_base) == Generic:
            return get_args(orig_base)
    raise ValueError(f"{cls} class does not have a `Generic` parent.")


def get_type_var_mapping(
    cls: type[Any],
) -> tuple[type[Any], dict[TypeVar, Any]]:
    """
    Given a class that specializes a generic class, return the generic class and
    the mapping from the generic class type variables to the specialized arguments.
    """

    if not issubclass(cls, Generic):
        raise ValueError(f"{cls} does not specialize a generic class.")

    # Get the generic parent
    orig_bases: Iterable[Any] = getattr(cls, "__orig_bases__")
    orig_bases = [
        orig_base
        for orig_base in orig_bases
        if (origin := get_origin(orig_base)) is not Generic and origin is not None
    ]
    # Do not handle more than one generic parent in the mro.
    # It is possible to handle more than one generic parent, but
    # the mapping of type variables will be more complex, especially for
    # generic parents inheriting from other generic parents.
    if len(orig_bases) != 1:
        raise ValueError(
            "Class cannot have more than one generic class in its mro. This "
            "restriction may be lifted in the future.",
            orig_bases,
        )

    # Get the generic operation, and its specialized type parameters.
    generic_parent: type[Any] = get_origin(orig_bases[0])
    specialized_args = get_args(orig_bases[0])

    # Get the `TypeVar` used in the generic parent
    generic_args = get_type_var_from_generic_class(generic_parent)

    if len(generic_args) != len(specialized_args):
        raise ValueError(
            f"Generic class {generic_parent} class has {len(generic_args)} "
            f"parameters, but {cls} specialize {len(specialized_args)} of them."
        )

    type_var_mapping = dict(zip(generic_args, specialized_args))
    return generic_parent, type_var_mapping


def type_repr(obj: Any) -> str:
    """Return the repr() of an object, special-casing types."""
    if isinstance(obj, types.GenericAlias):
        origin = get_origin(obj)
        args = get_args(obj)
        return f"{type_repr(origin)}[{', '.join(type_repr(arg) for arg in args)}]"
    if isinstance(obj, types.UnionType):
        args = get_args(obj)
        return f"{'|'.join(type_repr(arg) for arg in args)}"
    if obj is type(None):
        return "None"
    if obj is ...:
        return "..."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    if isinstance(obj, type):
        return f"{obj.__qualname__}"
    return repr(obj)
