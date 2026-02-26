import types
from collections.abc import Iterable, Sequence
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    TypeGuard,
    Union,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeVar

from xdsl.ir import ParametrizedAttribute, SSAValue
from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from typing_extensions import TypeForm

_T = TypeVar("_T")


def isa(arg: Any, hint: "TypeForm[_T]") -> TypeGuard[_T]:
    from xdsl.irdl import ConstraintContext

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

    if origin is SSAValue:
        if not isinstance(arg, SSAValue):
            return False
        arg = cast(SSAValue, arg)
        return isa(arg.type, get_args(hint)[0])

    raise ValueError(f"isa: unsupported type hint '{hint}' {get_origin(hint)}")


def assert_isa(arg: Any, hint: "TypeForm[_T]") -> TypeGuard[_T]:
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
) -> tuple[tuple[TypeVar, ...], dict[TypeVar, Any]]:
    """
    Given a Generic class, return the TypeVars used to specialize it, and the mapping
    from the generic class type variables to the specialized arguments for all type
    variables used in ancestor classes.
    Raises a ValueError if the specialized arguments for the same TypeVar are not
    consistent among superclasses.
    """

    if not issubclass(cls, Generic):
        raise ValueError(f"{cls} does not specialize a generic class.")

    orig_bases: Iterable[Any] = getattr(cls, "__orig_bases__")
    mapping: dict[TypeVar, Any] = {}
    args: tuple[TypeVar, ...] = ()

    for base in orig_bases:
        base_origin = get_origin(base)
        if base_origin is Generic:
            args = get_args(base)

    for base in orig_bases:
        base_origin = get_origin(base)
        if base_origin is Generic:
            continue
        base_cls = base if base_origin is None else base_origin
        if not issubclass(base_cls, Generic):
            continue
        base_type_vars, base_mapping = get_type_var_mapping(cast(type[Any], base_cls))
        base_args = get_args(base)

        for k, v in zip(base_type_vars, base_args, strict=True):
            if isinstance(v, TypeVar) and v not in args:
                # Pyright complains if there is a Generic parent that doesn't include
                # all the TypeVars used in later classes, so this is a valid check.
                raise ValueError(
                    f"Invalid definition {cls.__qualname__}, generic classes must subclass `Generic`."
                )
            if k is not v:
                # Don't assign forwarded TypeVars
                base_mapping[k] = v

        for k, v in base_mapping.items():
            if k in mapping:
                if v is not mapping[k]:
                    raise ValueError(
                        "Error extracting assignments of TypeVars of "
                        f"{cls.__qualname__}, inconsistent assignments to {k} in "
                        f"superclasses: {v.__qualname__}, {mapping[k].__qualname__}."
                    )
                continue
            mapping[k] = v

    return args, mapping


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
