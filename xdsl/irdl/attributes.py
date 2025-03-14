#  ____        _
# |  _ \  __ _| |_ __ _
# | | | |/ _` | __/ _` |
# | |_| | (_| | || (_| |
# |____/ \__,_|\__\__,_|
#

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from inspect import isclass
from types import FunctionType, GenericAlias, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from typing_extensions import TypeForm


from xdsl.ir import (
    Attribute,
    AttributeInvT,
    Data,
    ParametrizedAttribute,
    TypedAttribute,
)
from xdsl.utils.exceptions import PyRDLAttrDefinitionError, VerifyException
from xdsl.utils.hints import (
    PropertyType,
    get_type_var_from_generic_class,
)
from xdsl.utils.runtime_final import runtime_final

from .constraints import (  # noqa: TID251
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    ConstraintVar,
    EqAttrConstraint,
    GenericAttrConstraint,
    ParamAttrConstraint,
    VarConstraint,
)
from .error import IRDLAnnotations  # noqa: TID251

_DataElement = TypeVar("_DataElement", covariant=True)


@dataclass(frozen=True)
class GenericData(Data[_DataElement], ABC):
    """
    A Data with type parameters.
    """

    @staticmethod
    @abstractmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        """
        Given the generic parameters passed to the generic attribute type,
        return the corresponding attribute constraint.
        """


#  ____                              _   _   _
# |  _ \ __ _ _ __ __ _ _ __ ___    / \ | |_| |_ _ __
# | |_) / _` | '__/ _` | '_ ` _ \  / _ \| __| __| '__|
# |  __/ (_| | | | (_| | | | | | |/ ___ \ |_| |_| |
# |_|   \__,_|_|  \__,_|_| |_| |_/_/   \_\__|\__|_|
#

_A = TypeVar("_A", bound=Attribute)

ParameterDef = Annotated[_A, IRDLAnnotations.ParamDefAnnot]


def irdl_param_attr_get_param_type_hints(cls: type[_A]) -> list[tuple[str, Any]]:
    """Get the type hints of an IRDL parameter definitions."""
    res = list[tuple[str, Any]]()
    for field_name, field_type in get_type_hints(cls, include_extras=True).items():
        if field_name == "name" or field_name == "parameters":
            continue

        origin: Any | None = cast(Any | None, get_origin(field_type))
        args = get_args(field_type)
        if origin != Annotated or IRDLAnnotations.ParamDefAnnot not in args:
            raise PyRDLAttrDefinitionError(
                f"In attribute {cls.__name__} definition: Parameter "
                + f"definition {field_name} should be defined with "
                + f"type `ParameterDef[<Constraint>]`, got type {field_type}."
            )

        res.append((field_name, field_type))
    return res


_PARAMETRIZED_ATTRIBUTE_DICT_KEYS = {
    key
    for dict_seq in (
        (cls.__dict__ for cls in ParametrizedAttribute.mro()[::-1]),
        (Generic.__dict__, GenericData.__dict__),
    )
    for dict in dict_seq
    for key in dict
}


@dataclass
class ParamAttrDef:
    """The IRDL definition of a parametrized attribute."""

    name: str
    parameters: list[tuple[str, AttrConstraint]]

    @staticmethod
    def from_pyrdl(
        pyrdl_def: type[ParametrizedAttribute],
    ) -> "ParamAttrDef":
        # Get the fields from the class and its parents
        clsdict = {
            key: value
            for parent_cls in pyrdl_def.mro()[::-1]
            for key, value in parent_cls.__dict__.items()
            if key not in _PARAMETRIZED_ATTRIBUTE_DICT_KEYS
        }

        # Check that all fields of the attribute definition are either already
        # in ParametrizedAttribute, or are class functions or methods.
        for field_name, value in clsdict.items():
            if field_name == "name":
                continue
            if isinstance(
                value, FunctionType | PropertyType | classmethod | staticmethod
            ):
                continue
            # Constraint variables are allowed
            if get_origin(value) is Annotated:
                if any(isinstance(arg, ConstraintVar) for arg in get_args(value)):
                    continue
            raise PyRDLAttrDefinitionError(
                f"{field_name} is not a parameter definition."
            )

        if "name" not in clsdict:
            raise Exception(
                f"pyrdl attribute definition '{pyrdl_def.__name__}' does not "
                "define the attribute name. The attribute name is defined by "
                "adding a 'name' field."
            )

        name = clsdict["name"]

        param_hints = irdl_param_attr_get_param_type_hints(pyrdl_def)

        parameters = list[tuple[str, AttrConstraint]]()
        for param_name, param_type in param_hints:
            constraint = irdl_to_attr_constraint(param_type, allow_type_var=True)
            parameters.append((param_name, constraint))

        return ParamAttrDef(name, parameters)

    def verify(self, attr: ParametrizedAttribute):
        """Verify that `attr` satisfies the invariants."""

        if len(attr.parameters) != len(self.parameters):
            raise VerifyException(
                f"In {self.name} attribute verifier: "
                f"{len(self.parameters)} parameters expected, got "
                f"{len(attr.parameters)}"
            )
        constraint_context = ConstraintContext()
        for param, (_, param_def) in zip(attr.parameters, self.parameters):
            param_def.verify(param, constraint_context)


_PAttrTT = TypeVar("_PAttrTT", bound=type[ParametrizedAttribute])


def get_accessors_from_param_attr_def(attr_def: ParamAttrDef):
    # New fields and methods added to the attribute
    new_fields = dict[str, Any]()

    def param_name_field(idx: int):
        @property
        def field(self: ParametrizedAttribute):
            return self.parameters[idx]

        return field

    for idx, (param_name, _) in enumerate(attr_def.parameters):
        new_fields[param_name] = param_name_field(idx)

    @classmethod
    def get_irdl_definition(cls: type[ParametrizedAttribute]):
        return attr_def

    new_fields["get_irdl_definition"] = get_irdl_definition
    return new_fields


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


def irdl_attr_definition(cls: TypeAttributeInvT) -> TypeAttributeInvT:
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


IRDLGenericAttrConstraint: TypeAlias = (
    GenericAttrConstraint[AttributeInvT]
    | Attribute
    | type[AttributeInvT]
    | "TypeForm[AttributeInvT]"
    | ConstraintVar
    | TypeVar
)
"""
Attribute constraints represented using the IRDL python frontend. Attribute constraints
can either be:
- An instance of `AttrConstraint` representing a constraint on an attribute.
- An instance of `Attribute` representing an equality constraint on an attribute.
- A type representing a specific attribute class.
- A TypeForm that can represent both unions and generic attributes.
- A `ConstraintVar` representing a constraint variable.
"""

IRDLAttrConstraint = IRDLGenericAttrConstraint[Attribute]
"""See `IRDLGenericAttrConstraint`."""


def irdl_list_to_attr_constraint(
    pyrdl_constraints: Sequence[IRDLAttrConstraint],
    *,
    allow_type_var: bool = False,
    type_var_mapping: dict[TypeVar, AttrConstraint] | None = None,
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
                type_var_mapping=type_var_mapping,
            )
            return VarConstraint(arg.name, constraint)

    constraints: list[AttrConstraint] = []
    for arg in pyrdl_constraints:
        # We should not try to convert IRDL annotations, which do not
        # correspond to constraints
        if isinstance(arg, IRDLAnnotations):
            continue
        constraints.append(
            irdl_to_attr_constraint(
                arg,
                allow_type_var=allow_type_var,
                type_var_mapping=type_var_mapping,
            )
        )
    if len(constraints) == 0:
        return AnyAttr()
    if len(constraints) > 1:
        return AllOf(tuple(constraints))
    return constraints[0]


def irdl_to_attr_constraint(
    irdl: IRDLAttrConstraint,
    *,
    allow_type_var: bool = False,
    type_var_mapping: dict[TypeVar, AttrConstraint] | None = None,
) -> AttrConstraint:
    if isinstance(irdl, GenericAttrConstraint):
        return cast(AttrConstraint, irdl)

    if isinstance(irdl, Attribute):
        return EqAttrConstraint(irdl)

    # Annotated case
    # Each argument of the Annotated type corresponds to a constraint to satisfy.
    # If there is a `ConstraintVar` annotation, we add the entire constraint to
    # the constraint variable.
    if get_origin(irdl) == Annotated:
        return irdl_list_to_attr_constraint(
            get_args(irdl),
            allow_type_var=allow_type_var,
            type_var_mapping=type_var_mapping,
        )

    # Attribute class case
    # This is an `AnyAttr`, which does not constrain the attribute.
    if irdl is Attribute:
        return AnyAttr()

    # Attribute class case
    # This is a coercion for an `BaseAttr`.
    if (
        isclass(irdl)
        and not isinstance(irdl, GenericAlias)
        and issubclass(irdl, Attribute)
    ):
        return BaseAttr(irdl)

    # Type variable case
    # We take the type variable bound constraint.
    if isinstance(irdl, TypeVar):
        if not allow_type_var:
            raise Exception("TypeVar in unexpected context.")
        if type_var_mapping:
            if irdl in type_var_mapping:
                return type_var_mapping[irdl]
        if irdl.__bound__ is None:
            raise Exception("Type variables used in IRDL are expected to be bound.")
        # We do not allow nested type variables.
        return irdl_to_attr_constraint(irdl.__bound__)

    origin = get_origin(irdl)

    # GenericData case
    if isclass(origin) and issubclass(origin, GenericData):
        args = get_args(irdl)
        if len(args) != 1:
            raise Exception(f"GenericData args must have length 1, got {args}")
        origin = cast(type[GenericData[Any]], origin)
        args = cast(tuple[Attribute], args)
        return AllOf((BaseAttr(origin), origin.generic_constraint_coercion(args)))

    # Generic ParametrizedAttributes case
    # We translate it to constraints over the attribute parameters.
    if (
        isclass(origin)
        and issubclass(origin, ParametrizedAttribute)
        and issubclass(origin, Generic)
    ):
        args = [
            irdl_to_attr_constraint(
                arg, allow_type_var=allow_type_var, type_var_mapping=type_var_mapping
            )
            for arg in get_args(irdl)
        ]
        generic_args = get_type_var_from_generic_class(origin)

        # Check that we have the right number of parameters
        if len(args) != len(generic_args):
            raise Exception(
                f"{origin.name} expects {len(generic_args)}"
                f" parameters, got {len(args)}."
            )

        type_var_mapping = {
            parameter: arg for parameter, arg in zip(generic_args, args)
        }

        origin_parameters = irdl_param_attr_get_param_type_hints(origin)
        origin_constraints = [
            irdl_to_attr_constraint(
                param, allow_type_var=True, type_var_mapping=type_var_mapping
            )
            for _, param in origin_parameters
        ]
        return ParamAttrConstraint(origin, origin_constraints)

    # Union case
    # This is a coercion for an `AnyOf` constraint.
    if origin == UnionType or origin == Union:
        constraints: list[AttrConstraint] = []
        for arg in get_args(irdl):
            # We should not try to convert IRDL annotations, which do not
            # correspond to constraints
            if isinstance(arg, IRDLAnnotations):
                continue
            constraints.append(
                irdl_to_attr_constraint(
                    arg,
                    allow_type_var=allow_type_var,
                    type_var_mapping=type_var_mapping,
                )
            )
        if len(constraints) > 1:
            return AnyOf(constraints)
        return constraints[0]

    # Better error messages for missing GenericData in Data definitions
    if isclass(origin) and issubclass(origin, Data):
        raise ValueError(
            f"Generic `Data` type '{origin.name}' cannot be converted to "
            "an attribute constraint. Consider making it inherit from "
            "`GenericData` instead of `Data`."
        )

    raise ValueError(f"Unexpected irdl constraint: {irdl}")


def base(irdl: type[AttributeInvT]) -> GenericAttrConstraint[AttributeInvT]:
    """
    Converts an attribute type into the equivalent constraint, detecting generic
    parameters if present.
    """
    return cast(GenericAttrConstraint[AttributeInvT], irdl_to_attr_constraint(irdl))


def eq(irdl: AttributeInvT) -> GenericAttrConstraint[AttributeInvT]:
    """
    Converts an attribute instance into the equivalent constraint.
    """
    return cast(GenericAttrConstraint[AttributeInvT], irdl_to_attr_constraint(irdl))
