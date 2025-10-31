#  ____        _
# |  _ \  __ _| |_ __ _
# | | | |/ _` | __/ _` |
# | |_| | (_| | || (_| |
# |____/ \__,_|\__\__,_|
#

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from inspect import get_annotations, isclass
from types import FunctionType, GenericAlias, UnionType
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeAlias,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import TypeVar, dataclass_transform

if sys.version_info >= (3, 14, 0):
    from typing_extensions import TypeForm
else:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from typing_extensions import TypeForm


from xdsl.ir import (
    Attribute,
    AttributeCovT,
    AttributeInvT,
    BuiltinAttribute,
    Data,
    ParametrizedAttribute,
    TypedAttribute,
)
from xdsl.utils.classvar import is_const_classvar
from xdsl.utils.exceptions import PyRDLAttrDefinitionError, PyRDLTypeError
from xdsl.utils.hints import (
    PropertyType,
    get_type_var_from_generic_class,
)
from xdsl.utils.runtime_final import runtime_final

from .constraints import (  # noqa: TID251
    AllOf,
    AnyAttr,
    AnyInt,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    ConstraintVar,
    EqAttrConstraint,
    EqIntConstraint,
    IntConstraint,
    IntSetConstraint,
    IntTypeVarConstraint,
    ParamAttrConstraint,
    RangeConstraint,
    RangeOf,
    SingleOf,
    TypeVarConstraint,
    VarConstraint,
)

_DataElement = TypeVar("_DataElement", bound=Hashable, covariant=True)


class GenericData(Data[_DataElement], ABC):
    """
    A Data with type parameters.
    """

    @staticmethod
    @abstractmethod
    def constr() -> AttrConstraint:
        """
        Returns a constraint for this subclass.
        Generic arguments are constrained via TypeVarConstraints.
        """


#  ____                              _   _   _
# |  _ \ __ _ _ __ __ _ _ __ ___    / \ | |_| |_ _ __
# | |_) / _` | '__/ _` | '_ ` _ \  / _ \| __| __| '__|
# |  __/ (_| | | | (_| | | | | | |/ ___ \ |_| |_| |
# |_|   \__,_|_|  \__,_|_| |_| |_/_/   \_\__|\__|_|
#


class _ParameterDef:
    """
    Parameter field definition class for `@irdl_param_attr_definition`.
    """

    param: AttrConstraint | None
    converter: Callable[[Any], Attribute] | None

    def __init__(
        self,
        param: AttrConstraint | None,
        converter: Callable[[Any], Attribute] | None,
    ):
        self.param = param
        self.converter = converter


def param_def(
    constraint: AttrConstraint[AttributeInvT] | None = None,
    *,
    converter: Callable[[Any], AttributeInvT] | None = None,
    init: Literal[True] = True,
) -> AttributeInvT:
    """Defines a property of an operation."""
    return cast(AttributeInvT, _ParameterDef(constraint, converter))


def check_attr_name(cls: type):
    """Check that the attribute class has a correct name."""
    name = None
    for base in cls.mro():
        if "name" in base.__dict__:
            name = base.__dict__["name"]
            break

    if not isinstance(name, str):
        raise PyRDLAttrDefinitionError(
            f"pyrdl attribute definition '{cls.__name__}' does not "
            "define the attribute name. The attribute name is defined by "
            "adding a 'name' field with a string value."
        )

    dialect_attr_name = name.split(".")
    if len(dialect_attr_name) >= 2:
        return

    if not issubclass(cls, BuiltinAttribute):
        raise PyRDLAttrDefinitionError(
            f"Name '{name}' is not a valid attribute name. It should be of the form "
            "'<dialect>.<name>'."
        )


_PARAMETRIZED_ATTRIBUTE_DICT_KEYS = {
    key
    for dict_seq in (
        (cls.__dict__ for cls in ParametrizedAttribute.mro()[::-1]),
        (Generic.__dict__, GenericData.__dict__),
    )
    for dict in dict_seq
    for key in dict
}

_IGNORED_PARAM_ATTR_FIELD_TYPES = set(("name", "parameters"))


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class ParamDef(NamedTuple):
    """
    Contains information about a parameter,
    effectively acting as a resolved `_ParameterDef`
    """

    constr: AttrConstraint
    converter: Callable[[Any], Attribute] | None = None


@dataclass
class ParamAttrDef:
    """The IRDL definition of a parametrized attribute."""

    name: str
    parameters: list[tuple[str, ParamDef]]

    @staticmethod
    def from_pyrdl(
        pyrdl_def: type[ParametrizedAttribute],
    ) -> "ParamAttrDef":
        # Get the fields from the class and its parents
        clsdict = {
            key: value
            for parent_cls in pyrdl_def.mro()[::-1]
            for key, value in parent_cls.__dict__.items()
            if key not in _PARAMETRIZED_ATTRIBUTE_DICT_KEYS and not _is_dunder(key)
        }

        if "name" not in clsdict:
            raise Exception(
                f"pyrdl attribute definition '{pyrdl_def.__name__}' does not "
                "define the attribute name. The attribute name is defined by "
                "adding a 'name' field."
            )

        name = clsdict["name"]

        # Get type hints
        field_types = {
            field_name: field_type
            for parent_cls in pyrdl_def.mro()[::-1]
            for field_name, field_type in get_annotations(
                parent_cls, eval_str=True
            ).items()
            if field_name not in _IGNORED_PARAM_ATTR_FIELD_TYPES
        }

        # Get assigned values
        field_values = {
            field_name: field_value
            for field_name, field_value in clsdict.items()
            if (
                # Ignore name field
                field_name != "name"
                # Ignore functions
                and not isinstance(
                    field_value,
                    FunctionType | PropertyType | classmethod | staticmethod,
                )
            )
        }

        # The resulting parameters
        parameters: dict[str, ParamDef] = {}

        for field_name, field_type in field_types.items():
            if is_const_classvar(field_name, field_type, PyRDLAttrDefinitionError):
                field_values.pop(field_name, None)
                continue
            try:
                constraint = irdl_to_attr_constraint(field_type, allow_type_var=True)
            except TypeError as e:
                raise PyRDLAttrDefinitionError(
                    f"Invalid field type {field_type} for field name {field_name}."
                ) from e

            converter: Callable[[Any], Attribute] | None = None

            if field_name in field_values:
                value = field_values.pop(field_name)
                if isinstance(value, _ParameterDef):
                    try:
                        if value.param is not None:
                            constraint &= irdl_to_attr_constraint(
                                value.param, allow_type_var=True
                            )
                    except TypeError as e:
                        raise PyRDLAttrDefinitionError(
                            f"Invalid constraint {value.param} for field name {field_name}."
                        ) from e
                    if value.converter is not None:
                        converter = value.converter

                # Constraint variables are deprecated
                elif get_origin(value) is Annotated or any(
                    isinstance(arg, ConstraintVar) for arg in get_args(value)
                ):
                    import warnings

                    warnings.warn(
                        "The use of `ConstraintVar` is deprecated, please use `VarConstraint`",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    raise PyRDLAttrDefinitionError(
                        f"{field_name} is not a parameter definition."
                    )

            parameters[field_name] = ParamDef(constraint, converter)

        for field_name, value in field_values.items():
            # Anything left is a field without an annotation or a constaint var.
            if get_origin(value) is Annotated or any(
                isinstance(arg, ConstraintVar) for arg in get_args(value)
            ):
                import warnings

                warnings.warn(
                    "The use of `ConstraintVar` is deprecated, please use `VarConstraint`",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                raise PyRDLAttrDefinitionError(
                    f"Missing field type for parameter name {field_name}"
                )

        return ParamAttrDef(name, list(parameters.items()))

    def verify(self, attr: ParametrizedAttribute):
        """Verify that `attr` satisfies the invariants."""

        constraint_context = ConstraintContext()
        for field, param_def in self.parameters:
            param_def.constr.verify(getattr(attr, field), constraint_context)


_PAttrTT = TypeVar("_PAttrTT", bound=type[ParametrizedAttribute])


def get_accessors_from_param_attr_def(attr_def: ParamAttrDef) -> dict[str, Any]:
    @classmethod
    def get_irdl_definition(cls: type[ParametrizedAttribute]):
        return attr_def

    return {"get_irdl_definition": get_irdl_definition}


def irdl_param_attr_definition(cls: _PAttrTT) -> _PAttrTT:
    """Decorator used on classes to define a new attribute definition."""

    attr_def = ParamAttrDef.from_pyrdl(cls)
    new_fields = get_accessors_from_param_attr_def(attr_def)

    if issubclass(cls, TypedAttribute):
        type_indexes = tuple(
            i for i, (p, _) in enumerate(attr_def.parameters) if p == "type"
        )
        if not type_indexes:
            raise PyRDLAttrDefinitionError(
                f"TypedAttribute {cls.__name__} should have a 'type' parameter."
            )
        type_index = type_indexes[0]

        @classmethod
        def get_type_index(cls: Any) -> int:
            return type_index

        new_fields["get_type_index"] = get_type_index

    return runtime_final(
        dataclass(frozen=True, init=False)(
            type.__new__(
                type(cls),
                cls.__name__,
                (cls,),
                {**cls.__dict__, **new_fields},
            )
        )
    )


TypeAttributeInvT = TypeVar("TypeAttributeInvT", bound=type[Attribute])


@overload
def irdl_attr_definition(
    cls: TypeAttributeInvT, *, init: bool = True
) -> TypeAttributeInvT: ...


@overload
def irdl_attr_definition(
    *, init: bool = True
) -> Callable[[TypeAttributeInvT], TypeAttributeInvT]: ...


@dataclass_transform(frozen_default=True, field_specifiers=(param_def,))
def irdl_attr_definition(
    cls: TypeAttributeInvT | None = None, *, init: bool = True
) -> TypeAttributeInvT | Callable[[TypeAttributeInvT], TypeAttributeInvT]:
    def decorator(cls: TypeAttributeInvT) -> TypeAttributeInvT:
        check_attr_name(cls)
        if issubclass(cls, ParametrizedAttribute):
            return irdl_param_attr_definition(cls)
        if issubclass(cls, Data):
            # This used to be convoluted
            # But Data is already frozen itself, so any child Attribute still throws on
            # .data!
            return runtime_final(cast(TypeAttributeInvT, cls))
        raise TypeError(
            f"Class {cls.__name__} should either be a subclass of 'Data' or "
            "'ParametrizedAttribute'"
        )

    if cls is None:
        return decorator
    return decorator(cls)


# Cannot subclass ABC as it conflicts with Enum's metaclass.
class ConstraintConvertible(Generic[AttributeCovT]):
    """
    Abstract superclass for values that have corresponding Attribute constraints.
    """

    @staticmethod
    @abstractmethod
    def base_constr() -> AttrConstraint[AttributeCovT]:
        """The constraint for this class."""

    @abstractmethod
    def constr(self) -> AttrConstraint[AttributeCovT]:
        """The constraint for this instance."""


# Dynamic check due to a regression in Python 3.14, which broke `|` between types and
# strings, and a bug in Marimo where the correct version of typing_extensions is not
# installed due to an old (4.11.0) version of typing_extensions already being present in
# the Pyodide build.
# When Marimo update their Pyodide we should delete the second branch.
# We will likely want to support 3.14.0 for a long time, so we can't remove the first
# branch even if the `|` bug is fixed in a patch update.
if sys.version_info >= (3, 14, 0):
    IRDLAttrConstraint: TypeAlias = (
        AttrConstraint[AttributeInvT]
        | AttributeInvT
        | type[AttributeInvT]
        | type[ConstraintConvertible[AttributeInvT]]
        | ConstraintConvertible[AttributeInvT]
        | TypeForm[AttributeInvT]
    )
    """
    Attribute constraints represented using the IRDL python frontend. Attribute constraints
    can either be:
    - An instance of `AttrConstraint` representing a constraint on an attribute.
    - An instance of `Attribute` representing an equality constraint on an attribute.
    - A type representing a specific attribute class.
    - A TypeForm that can represent both unions and generic attributes.
    - An instance or subclass of ConstraintConvertible.
    """
else:
    IRDLAttrConstraint: TypeAlias = (
        AttrConstraint[AttributeInvT]
        | AttributeInvT
        | type[AttributeInvT]
        | type[ConstraintConvertible[AttributeInvT]]
        | ConstraintConvertible[AttributeInvT]
        | "TypeForm[AttributeInvT]"
    )
    """
    Attribute constraints represented using the IRDL python frontend. Attribute constraints
    can either be:
    - An instance of `AttrConstraint` representing a constraint on an attribute.
    - An instance of `Attribute` representing an equality constraint on an attribute.
    - A type representing a specific attribute class.
    - A TypeForm that can represent both unions and generic attributes.
    - An instance or subclass of ConstraintConvertible.
    """


def irdl_list_to_attr_constraint(
    pyrdl_constraints: Sequence[IRDLAttrConstraint],
    *,
    allow_type_var: bool = False,
) -> AttrConstraint:
    """
    Convert a list of PyRDL type annotations to an AttrConstraint.
    Each list element correspond to a constraint to satisfy.
    If there is a `ConstraintVar` annotation, we add the entire constraint to
    the constraint variable.
    """
    # Check for a constraint varibale first
    for idx, arg in enumerate(pyrdl_constraints):
        if isinstance(arg, ConstraintVar):
            constraint = irdl_list_to_attr_constraint(
                list(pyrdl_constraints[:idx]) + list(pyrdl_constraints[idx + 1 :]),
                allow_type_var=allow_type_var,
            )
            return VarConstraint(arg.name, constraint)

    constraints = tuple(
        irdl_to_attr_constraint(arg, allow_type_var=allow_type_var)
        for arg in pyrdl_constraints
    )

    if len(constraints) > 1:
        return AllOf(constraints)

    if not constraints:
        return AnyAttr()

    return constraints[0]


def irdl_to_attr_constraint(
    irdl: IRDLAttrConstraint[AttributeInvT],
    *,
    allow_type_var: bool = False,
) -> AttrConstraint[AttributeInvT]:
    if isinstance(irdl, AttrConstraint):
        return cast(AttrConstraint[AttributeInvT], irdl)

    if isinstance(irdl, Attribute):
        return cast(AttrConstraint[AttributeInvT], EqAttrConstraint(irdl))

    if isinstance(irdl, ConstraintConvertible):
        value = cast(ConstraintConvertible[AttributeInvT], irdl)
        return value.constr()

    # Annotated case
    # Each argument of the Annotated type corresponds to a constraint to satisfy.
    # If there is a `ConstraintVar` annotation, we add the entire constraint to
    # the constraint variable.
    if get_origin(irdl) == Annotated:
        return cast(
            AttrConstraint[AttributeInvT],
            irdl_list_to_attr_constraint(
                get_args(irdl),
                allow_type_var=allow_type_var,
            ),
        )

    # Attribute class case
    # This is an `AnyAttr`, which does not constrain the attribute.
    if irdl is Attribute:
        return cast(AttrConstraint[AttributeInvT], AnyAttr())

    # Attribute class case
    # This is a coercion for an `BaseAttr`.
    if (
        isclass(irdl)
        and not isinstance(irdl, GenericAlias)
        and issubclass(irdl, Attribute)
    ):
        return cast(AttrConstraint[AttributeInvT], BaseAttr(irdl))

    # Type variable case
    # We take the type variable bound constraint.
    if isinstance(irdl, TypeVar):
        if not allow_type_var:
            raise PyRDLTypeError("TypeVar in unexpected context.")
        if irdl.__bound__ is None:
            raise PyRDLTypeError(
                "Type variables used in IRDL are expected to be bound."
            )
        # We do not allow nested type variables.
        constraint = irdl_to_attr_constraint(irdl.__bound__)
        return cast(AttrConstraint[AttributeInvT], TypeVarConstraint(irdl, constraint))

    origin = get_origin(irdl)

    # Generic case
    if isclass(origin) and (
        issubclass(origin, GenericData)
        or (issubclass(origin, ParametrizedAttribute) and issubclass(origin, Generic))
    ):
        if issubclass(origin, GenericData):
            base_constr = origin.constr()
        else:
            base_constr = ParamAttrConstraint(
                origin,
                [param.constr for _, param in origin.get_irdl_definition().parameters],
            )

        type_vars = get_type_var_from_generic_class(cast(type, origin))
        args: tuple[IRDLAttrConstraint | int | TypeForm[int], ...] = get_args(irdl)

        # Here, we use `__default__` to check for default values, instead of `has_default`.
        # This is so users that are not using typing-extensions can still use it, as well
        # as marimo notebooks, which seems to replace `typing-extensions` with `typing`.
        if len(args) < len(type_vars) and all(
            hasattr(arg, "__default__") for arg in type_vars[len(args) :]
        ):
            # Check for default values
            args += tuple(arg.__default__ for arg in type_vars[len(args) :])

        # Check that we have the right number of parameters
        if len(args) != len(type_vars):
            raise PyRDLTypeError(
                f"{origin.name} expects {len(type_vars)} parameters, got {len(args)}."
            )

        type_var_mapping = {
            type_var: get_constraint(arg, allow_type_var=allow_type_var)
            for type_var, arg in zip(type_vars, args)
        }

        return cast(
            AttrConstraint[AttributeInvT],
            base_constr.mapping_type_vars(type_var_mapping),
        )

    # Union case
    # This is a coercion for an `AnyOf` constraint.
    if origin == UnionType or origin == Union:
        constraints: list[AttrConstraint] = []
        for arg in get_args(irdl):
            constraints.append(
                irdl_to_attr_constraint(
                    arg,
                    allow_type_var=allow_type_var,
                )
            )
        if len(constraints) > 1:
            return cast(AttrConstraint[AttributeInvT], AnyOf(constraints))
        return cast(AttrConstraint[AttributeInvT], constraints[0])

    if isclass(irdl) and issubclass(irdl, ConstraintConvertible):
        attr_data = cast(type[ConstraintConvertible[AttributeInvT]], irdl)
        return attr_data.base_constr()

    if origin is Literal:
        literal_args = get_args(irdl)
        if len(literal_args) == 1:
            return irdl_to_attr_constraint(literal_args[0])
        return AnyOf(literal_args)

    # Better error messages for missing GenericData in Data definitions
    if isclass(origin) and issubclass(origin, Data):
        raise PyRDLTypeError(
            f"Generic `Data` type '{origin.name}' cannot be converted to "
            "an attribute constraint. Consider making it inherit from "
            "`GenericData` instead of `Data`."
        )

    raise PyRDLTypeError(f"Unexpected irdl constraint: {irdl}")


def base(irdl: type[AttributeInvT]) -> AttrConstraint[AttributeInvT]:
    """
    Converts an attribute type into the equivalent constraint, detecting generic
    parameters if present.
    """
    return irdl_to_attr_constraint(irdl)


def eq(irdl: AttributeInvT) -> AttrConstraint[AttributeInvT]:
    """
    Converts an attribute instance into the equivalent constraint.
    """
    return irdl_to_attr_constraint(irdl)


def get_optional_int_constraint(arg: Any) -> IntConstraint | None:
    """
    If the input is an int or an int type, return the corresponding constraint,
    otherwise return None.
    """
    if isinstance(arg, int):
        return EqIntConstraint(arg)

    if isclass(arg) and issubclass(arg, int):
        return AnyInt()

    if get_origin(arg) is Literal:
        literal_args = get_args(arg)

        if all(isinstance(literal_arg, int) for literal_arg in literal_args):
            if len(literal_args) == 1:
                return EqIntConstraint(literal_args[0])
            else:
                ints = frozenset(literal_arg for literal_arg in literal_args)
                return IntSetConstraint(ints)

    if get_origin(arg) is Union:
        union_args = get_args(arg)
        if all(
            (get_origin(union_arg) is Literal)
            and all(isinstance(literal_arg, int) for literal_arg in get_args(union_arg))
            for union_arg in union_args
        ):
            ints = frozenset(
                literal_arg
                for union_arg in union_args
                for literal_arg in get_args(union_arg)
            )
            return IntSetConstraint(ints)

    if (
        isinstance(arg, TypeVar)
        and (base := get_optional_int_constraint(arg.__bound__)) is not None
    ):
        return IntTypeVarConstraint(arg, base)


def get_int_constraint(arg: "int | TypeForm[int]") -> IntConstraint:
    """
    If the input is an int or an int type, return the corresponding constraint,
    otherwise raise `PyRDLTypeError`.
    """
    if (ic := get_optional_int_constraint(arg)) is not None:
        return ic

    raise PyRDLTypeError(f"Unexpected int type: {arg}")


def range_constr_coercion(
    attr: (
        AttributeCovT
        | type[AttributeCovT]
        | AttrConstraint[AttributeCovT]
        | RangeConstraint[AttributeCovT]
    ),
) -> RangeConstraint[AttributeCovT]:
    if isinstance(attr, RangeConstraint):
        return attr
    return RangeOf(irdl_to_attr_constraint(attr))


def single_range_constr_coercion(
    attr: AttributeCovT | type[AttributeCovT] | AttrConstraint[AttributeCovT],
) -> RangeConstraint[AttributeCovT]:
    return SingleOf(irdl_to_attr_constraint(attr))


@overload
def get_constraint(
    arg: "int | TypeForm[int]",
    *,
    allow_type_var: bool = False,
) -> IntConstraint: ...


@overload
def get_constraint(
    arg: IRDLAttrConstraint,
    *,
    allow_type_var: bool = False,
) -> AttrConstraint: ...


@overload
def get_constraint(
    arg: "IRDLAttrConstraint | int | TypeForm[int]",
    *,
    allow_type_var: bool = False,
) -> AttrConstraint | IntConstraint: ...


def get_constraint(
    arg: "IRDLAttrConstraint | int | TypeForm[int]",
    *,
    allow_type_var: bool = False,
) -> AttrConstraint | IntConstraint:
    """
    Converts the input expression, constraint, or type to the corresponding constraint.
    """
    if (ic := get_optional_int_constraint(arg)) is not None:
        return ic

    return irdl_to_attr_constraint(arg, allow_type_var=allow_type_var)  # pyright: ignore[reportArgumentType]
