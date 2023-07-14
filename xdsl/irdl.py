from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from inspect import isclass
from types import FunctionType, GenericAlias, UnionType
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Operation,
    OpResult,
    OpTrait,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.exceptions import (
    PyRDLAttrDefinitionError,
    PyRDLOpDefinitionError,
    VerifyException,
)
from xdsl.utils.hints import (
    PropertyType,
    get_type_var_from_generic_class,
    get_type_var_mapping,
)

# pyright: reportMissingParameterType=false, reportUnknownParameterType=false


def error(op: Operation, msg: str, e: Exception):
    diag = Diagnostic()
    diag.add_message(op, msg)
    diag.raise_exception(f"{op.name} operation does not verify", op, type(e), e)


class IRDLAnnotations(Enum):
    ParamDefAnnot = 1
    AttributeDefAnnot = 2
    OptAttributeDefAnnot = 3
    SingleBlockRegionAnnot = 4
    ConstraintVarAnnot = 5


#   ____                _             _       _
#  / ___|___  _ __  ___| |_ _ __ __ _(_)_ __ | |_ ___
# | |   / _ \| '_ \/ __| __| '__/ _` | | '_ \| __/ __|
# | |__| (_) | | | \__ \ |_| | | (_| | | | | | |_\__ \
#  \____\___/|_| |_|___/\__|_|  \__,_|_|_| |_|\__|___/
#


@dataclass
class AttrConstraint(ABC):
    """Constrain an attribute to a certain value."""

    @abstractmethod
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        """
        Check if the attribute satisfies the constraint,
        or raise an exception otherwise.
        """
        ...


@dataclass
class VarConstraint(AttrConstraint):
    """
    Constraint variable. If the variable is already set, this will constrain
    the attribute to be equal to the variable. Otherwise, it will first check that the
    variable satisfies the variable constraint, then set the variable with the
    attribute.
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""

    constraint: AttrConstraint
    """The constraint that the variable must satisfy."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if self.name in constraint_vars:
            if attr != constraint_vars[self.name]:
                raise VerifyException(
                    f"attribute {constraint_vars[self.name]} expected from variable "
                    f"'{self.name}', but got {attr}"
                )
        else:
            self.constraint.verify(attr, constraint_vars)
            constraint_vars[self.name] = attr


@dataclass
class ConstraintVar:
    """
    Annotation used in PyRDL to define a constraint variable.
    For instance, the following code defines a constraint variable T,
    that can then be used in PyRDL:
    ```python
    T = Annotated[PyRDLConstraint, ConstraintVar("T")]
    ```
    """

    name: str
    """The variable name. All uses of that name refer to the same variable."""


@dataclass
class EqAttrConstraint(AttrConstraint):
    """Constrain an attribute to be equal to another attribute."""

    attr: Attribute
    """The attribute we want to check equality with."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if attr != self.attr:
            raise VerifyException(f"Expected attribute {self.attr} but got {attr}")


@dataclass
class BaseAttr(AttrConstraint):
    """Constrain an attribute to be of a given base type."""

    attr: type[Attribute]
    """The expected attribute base type."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, self.attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.attr.name}"
            )


def attr_constr_coercion(
    attr: (Attribute | type[Attribute] | AttrConstraint),
) -> AttrConstraint:
    """
    Attributes are coerced into EqAttrConstraints,
    and Attribute types are coerced into BaseAttr.
    """
    if isinstance(attr, AttrConstraint):
        return attr
    if isinstance(attr, Attribute):
        return EqAttrConstraint(attr)
    if isclass(attr) and issubclass(attr, Attribute):
        return BaseAttr(attr)
    assert False


@dataclass
class AnyAttr(AttrConstraint):
    """Constraint that is verified by all attributes."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        pass


@dataclass(init=False)
class AnyOf(AttrConstraint):
    """Ensure that an attribute satisfies one of the given constraints."""

    attr_constrs: list[AttrConstraint]
    """The list of constraints that are checked."""

    def __init__(
        self, attr_constrs: Sequence[Attribute | type[Attribute] | AttrConstraint]
    ):
        self.attr_constrs = [attr_constr_coercion(constr) for constr in attr_constrs]

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        for attr_constr in self.attr_constrs:
            # Copy the constraint to ensure that if the constraint fails, the
            # constraint_vars are not modified.
            constraint_vars_copy = constraint_vars.copy()
            try:
                attr_constr.verify(attr, constraint_vars_copy)
                # If the constraint succeeds, we update back the constraint variables
                constraint_vars.update(constraint_vars_copy)
                return
            except VerifyException:
                pass
        raise VerifyException(f"Unexpected attribute {attr}")


@dataclass()
class AllOf(AttrConstraint):
    """Ensure that an attribute satisfies all the given constraints."""

    attr_constrs: list[AttrConstraint]
    """The list of constraints that are checked."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        exc_bucket: list[VerifyException] = []

        for attr_constr in self.attr_constrs:
            try:
                attr_constr.verify(attr, constraint_vars)
            except VerifyException as e:
                exc_bucket.append(e)

        if len(exc_bucket):
            if len(exc_bucket) == 1:
                raise VerifyException(str(exc_bucket[0])) from exc_bucket[0]
            exc_msg = "The following constraints were not satisfied:\n"
            exc_msg += "\n".join([str(e) for e in exc_bucket])
            raise VerifyException(exc_msg)


@dataclass(init=False)
class ParamAttrConstraint(AttrConstraint):
    """
    Constrain an attribute to be of a given type,
    and also constrain its parameters with additional constraints.
    """

    base_attr: type[ParametrizedAttribute]
    """The base attribute type."""

    param_constrs: list[AttrConstraint]
    """The attribute parameter constraints"""

    def __init__(
        self,
        base_attr: type[ParametrizedAttribute],
        param_constrs: Sequence[(Attribute | type[Attribute] | AttrConstraint)],
    ):
        self.base_attr = base_attr
        self.param_constrs = [attr_constr_coercion(constr) for constr in param_constrs]

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, self.base_attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.base_attr.name}"
            )
        if len(self.param_constrs) != len(attr.parameters):
            raise VerifyException(
                f"{len(self.param_constrs)} parameters expected, "
                f"but got {len(attr.parameters)}"
            )
        for idx, param_constr in enumerate(self.param_constrs):
            param_constr.verify(attr.parameters[idx], constraint_vars)


def _irdl_list_to_attr_constraint(
    pyrdl_constraints: Sequence[Any],
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
            constraint = _irdl_list_to_attr_constraint(
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
        return AllOf(constraints)
    return constraints[0]


def irdl_to_attr_constraint(
    irdl: Any,
    *,
    allow_type_var: bool = False,
    type_var_mapping: dict[TypeVar, AttrConstraint] | None = None,
) -> AttrConstraint:
    if isinstance(irdl, AttrConstraint):
        return irdl

    if isinstance(irdl, Attribute):
        return EqAttrConstraint(irdl)

    # Annotated case
    # Each argument of the Annotated type corresponds to a constraint to satisfy.
    # If there is a `ConstraintVar` annotation, we add the entire constraint to
    # the constraint variable.
    if get_origin(irdl) == Annotated:
        return _irdl_list_to_attr_constraint(
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
            raise Exception("Type variables used in IRDL are expected to" " be bound.")
        # We do not allow nested type variables.
        return irdl_to_attr_constraint(irdl.__bound__)

    origin = get_origin(irdl)

    # GenericData case
    if isclass(origin) and issubclass(origin, GenericData):
        return AllOf(
            [BaseAttr(origin), origin.generic_constraint_coercion(get_args(irdl))]
        )

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


#   ___                       _   _
#  / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __
# | | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \
# | |_| | |_) |  __/ | | (_| | |_| | (_) | | | |
#  \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|
#       |_|

_OpT = TypeVar("_OpT", bound="IRDLOperation")


class IRDLOperation(Operation):
    def __init__(
        self: IRDLOperation,
        operands: Sequence[SSAValue | Operation | Sequence[SSAValue | Operation] | None]
        | None = None,
        result_types: Sequence[Attribute | Sequence[Attribute] | None] | None = None,
        attributes: Mapping[str, Attribute | None] | None = None,
        successors: Sequence[Block | Sequence[Block] | None] | None = None,
        regions: Sequence[
            Region
            | None
            | Sequence[Operation]
            | Sequence[Block]
            | Sequence[Region | Sequence[Operation] | Sequence[Block]]
        ]
        | None = None,
    ):
        if operands is None:
            operands = []
        if result_types is None:
            result_types = []
        if attributes is None:
            attributes = {}
        if successors is None:
            successors = []
        if regions is None:
            regions = []
        irdl_op_init(
            self,
            self.irdl_definition,
            operands,
            result_types,
            attributes,
            successors,
            regions,
        )

    @classmethod
    def build(
        cls: type[_OpT],
        operands: Sequence[SSAValue | Operation | Sequence[SSAValue | Operation] | None]
        | None = None,
        result_types: Sequence[Attribute | Sequence[Attribute] | None] | None = None,
        attributes: Mapping[str, Attribute | None] | None = None,
        successors: Sequence[Block | Sequence[Block] | None] | None = None,
        regions: Sequence[
            Region
            | None
            | Sequence[Operation]
            | Sequence[Block]
            | Sequence[Region | Sequence[Operation] | Sequence[Block]]
        ]
        | None = None,
    ) -> _OpT:
        """Create a new operation using builders."""
        op = cls.__new__(cls)
        IRDLOperation.__init__(
            op,
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            successors=successors,
            regions=regions,
        )
        return op

    @classmethod
    @property
    def irdl_definition(cls) -> OpDef:
        """Get the IRDL operation definition."""
        ...


@dataclass
class IRDLOption(ABC):
    """Additional option used in IRDL."""

    ...


@dataclass
class AttrSizedOperandSegments(IRDLOption):
    """
    Expect an attribute on the op that contains
    the sizes of the variadic operands.
    """

    attribute_name = "operand_segment_sizes"
    """Name of the attribute containing the variadic operand sizes."""


@dataclass
class AttrSizedResultSegments(IRDLOption):
    """
    Expect an attribute on the operation that contains
    the sizes of the variadic results.
    """

    attribute_name = "result_segment_sizes"
    """Name of the attribute containing the variadic result sizes."""


@dataclass
class AttrSizedRegionSegments(IRDLOption):
    """
    Expect an attribute on the op that contains
    the sizes of the variadic regions.
    """

    attribute_name = "region_segment_sizes"
    """Name of the attribute containing the variadic region sizes."""


@dataclass
class AttrSizedSuccessorSegments(IRDLOption):
    """
    Expect an attribute on the op that contains
    the sizes of the variadic successors.
    """

    attribute_name = "successor_segment_sizes"
    """Name of the attribute containing the variadic successor sizes."""


@dataclass
class OperandOrResultDef(ABC):
    """An operand or a result definition. Should not be used directly."""

    ...


@dataclass
class VariadicDef(OperandOrResultDef):
    """A variadic operand or result definition. Should not be used directly."""

    ...


@dataclass
class OptionalDef(VariadicDef):
    """An optional operand or result definition. Should not be used directly."""

    ...


@dataclass(init=False)
class OperandDef(OperandOrResultDef):
    """An IRDL operand definition."""

    constr: AttrConstraint
    """The operand constraint."""

    def __init__(self, attr: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(attr)


Operand: TypeAlias = SSAValue


@dataclass(init=False)
class VarOperandDef(OperandDef, VariadicDef):
    """An IRDL variadic operand definition."""


VarOperand: TypeAlias = list[SSAValue]


@dataclass(init=False)
class OptOperandDef(VarOperandDef, OptionalDef):
    """An IRDL optional operand definition."""


OptOperand: TypeAlias = SSAValue | None


@dataclass(init=False)
class ResultDef(OperandOrResultDef):
    """An IRDL result definition."""

    constr: AttrConstraint
    """The result constraint."""

    def __init__(self, attr: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(attr)


@dataclass(init=False)
class VarResultDef(ResultDef, VariadicDef):
    """An IRDL variadic result definition."""


VarOpResult: TypeAlias = list[OpResult]


@dataclass(init=False)
class OptResultDef(VarResultDef, OptionalDef):
    """An IRDL optional result definition."""


OptOpResult: TypeAlias = OpResult | None


@dataclass(init=True)
class RegionDef(Region):
    """
    An IRDL region definition.
    """


@dataclass
class VarRegionDef(RegionDef, VariadicDef):
    """An IRDL variadic region definition."""


@dataclass
class OptRegionDef(RegionDef, OptionalDef):
    """An IRDL optional region definition."""


VarRegion: TypeAlias = list[Region]
OptRegion: TypeAlias = Region | None


@dataclass
class SingleBlockRegionDef(RegionDef):
    """An IRDL region definition that expects exactly one block."""


class VarSingleBlockRegionDef(RegionDef, VariadicDef):
    """An IRDL variadic region definition that expects exactly one block."""


class OptSingleBlockRegionDef(RegionDef, OptionalDef):
    """An IRDL optional region definition that expects exactly one block."""


SingleBlockRegion: TypeAlias = Annotated[Region, IRDLAnnotations.SingleBlockRegionAnnot]
VarSingleBlockRegion: TypeAlias = Annotated[
    list[Region], IRDLAnnotations.SingleBlockRegionAnnot
]
OptSingleBlockRegion: TypeAlias = Annotated[
    Region | None, IRDLAnnotations.SingleBlockRegionAnnot
]


@dataclass(init=False)
class AttributeDef:
    """An IRDL attribute definition."""

    constr: AttrConstraint
    """The attribute constraint."""

    attr_name: str | None = None
    """The attribute name, in case it is different from the field name."""

    def __init__(
        self,
        attr: Attribute | type[Attribute] | AttrConstraint,
        attr_name: str | None = None,
    ):
        self.constr = attr_constr_coercion(attr)
        self.attr_name = attr_name


@dataclass(init=False)
class OptAttributeDef(AttributeDef):
    """An IRDL attribute definition for an optional attribute."""

    def __init__(
        self,
        attr: Attribute | type[Attribute] | AttrConstraint,
        attr_name: str | None = None,
    ):
        super().__init__(attr, attr_name=attr_name)


class SuccessorDef:
    """An IRDL successor definition."""


class VarSuccessorDef(SuccessorDef, VariadicDef):
    """An IRDL variadic successor definition."""


class OptSuccessorDef(SuccessorDef, OptionalDef):
    """An IRDL optional successor definition."""


Successor: TypeAlias = Block
OptSuccessor: TypeAlias = Block | None
VarSuccessor: TypeAlias = list[Block]

_ClsT = TypeVar("_ClsT")

# Field definition classes for `@irdl_op_definition`
# They carry the type information exactly as passed in the argument to `operand_def` etc.
# We can only convert them to constraints when creating the OpDef to allow for type var
# mapping.


class _OpDefField(Generic[_ClsT]):
    cls: type[_ClsT]

    def __init__(self, cls: type[_ClsT]):
        self.cls = cls


class _ConstrainedOpDefField(Generic[_ClsT], _OpDefField[_ClsT]):
    param: AttrConstraint | Attribute | type[Attribute] | TypeVar

    def __init__(
        self,
        cls: type[_ClsT],
        param: AttrConstraint | Attribute | type[Attribute] | TypeVar,
    ):
        super().__init__(cls)
        self.param = param


class _OperandFieldDef(_ConstrainedOpDefField[OperandDef,]):
    pass


class _ResultFieldDef(_ConstrainedOpDefField[ResultDef]):
    pass


class _AttributeFieldDef(_ConstrainedOpDefField[AttributeDef]):
    attr_name: str | None = None
    """The name of the attribute, in case it is different from the field name."""

    def __init__(
        self,
        cls: type[AttributeDef],
        param: AttrConstraint | Attribute | type[Attribute] | TypeVar,
        attr_name: str | None = None,
    ):
        super().__init__(cls, param)
        self.attr_name = attr_name


class _RegionFieldDef(_OpDefField[RegionDef]):
    pass


class _SuccessorFieldDef(_OpDefField[SuccessorDef]):
    pass


def result_def(
    constraint: AttrConstraint | Attribute | type[Attribute] | TypeVar = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OpResult:
    """
    Defines a result of an operation.
    """
    return cast(OpResult, _ResultFieldDef(ResultDef, constraint))


def var_result_def(
    constraint: AttrConstraint | Attribute | type[Attribute] | TypeVar = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarOpResult:
    """
    Defines a variadic result of an operation.
    """
    return cast(VarOpResult, _ResultFieldDef(VarResultDef, constraint))


def opt_result_def(
    constraint: AttrConstraint | Attribute | type[Attribute] | TypeVar = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptOpResult:
    """
    Defines an optional result of an operation.
    """
    return cast(OptOpResult, _ResultFieldDef(OptResultDef, constraint))


def attr_def(
    constraint: type[_AttrT] | TypeVar,
    *,
    attr_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> _AttrT:
    """
    Defines an attribute of an operation.
    """
    return cast(_AttrT, _AttributeFieldDef(AttributeDef, constraint, attr_name))


def opt_attr_def(
    constraint: type[_AttrT] | TypeVar,
    *,
    attr_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> _AttrT | None:
    """
    Defines an optional attribute of an operation.
    """
    return cast(_AttrT, _AttributeFieldDef(OptAttributeDef, constraint, attr_name))


def operand_def(
    constraint: AttrConstraint | Attribute | type[Attribute] | TypeVar = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> Operand:
    """
    Defines an operand of an operation.
    """
    return cast(Operand, _OperandFieldDef(OperandDef, constraint))


def var_operand_def(
    constraint: AttrConstraint | Attribute | type[Attribute] | TypeVar = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarOperand:
    """
    Defines a variadic operand of an operation.
    """
    return cast(VarOperand, _OperandFieldDef(VarOperandDef, constraint))


def opt_operand_def(
    constraint: AttrConstraint | Attribute | type[Attribute] | TypeVar = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptOperand:
    """
    Defines an optional operand of an operation.
    """
    return cast(OptOperand, _OperandFieldDef(OptOperandDef, constraint))


def region_def(
    single_block: Literal["single_block"] | None = None,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> Region:
    """
    Defines a region of an operation.
    """
    cls = RegionDef if single_block is None else SingleBlockRegionDef
    return cast(Region, _RegionFieldDef(cls))


def var_region_def(
    single_block: Literal["single_block"] | None = None,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarRegion:
    """
    Defines a variadic region of an operation.
    """
    cls = VarRegionDef if single_block is None else VarSingleBlockRegionDef
    return cast(VarRegion, _RegionFieldDef(cls))


def opt_region_def(
    single_block: Literal["single_block"] | None = None,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptRegion:
    """
    Defines an optional region of an operation.
    """
    cls = OptRegionDef if single_block is None else OptSingleBlockRegionDef
    return cast(OptRegion, _RegionFieldDef(cls))


def successor_def(
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> Successor:
    """
    Defines a successor of an operation.
    """
    return cast(Successor, _SuccessorFieldDef(SuccessorDef))


def var_successor_def(
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarSuccessor:
    """
    Defines a variadic successor of an operation.
    """
    return cast(VarSuccessor, _SuccessorFieldDef(VarSuccessorDef))


def opt_successor_def(
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptSuccessor:
    """
    Defines an optional successor of an operation.
    """
    return cast(OptSuccessor, _SuccessorFieldDef(OptSuccessorDef))


# Exclude `object`
_OPERATION_DICT_KEYS = {key for cls in Operation.mro()[:-1] for key in cls.__dict__}


@dataclass(kw_only=True)
class OpDef:
    """The internal IRDL definition of an operation."""

    name: str = field(kw_only=False)
    operands: list[tuple[str, OperandDef]] = field(default_factory=list)
    results: list[tuple[str, ResultDef]] = field(default_factory=list)
    attributes: dict[str, AttributeDef] = field(default_factory=dict)
    regions: list[tuple[str, RegionDef]] = field(default_factory=list)
    successors: list[tuple[str, SuccessorDef]] = field(default_factory=list)
    options: list[IRDLOption] = field(default_factory=list)
    traits: frozenset[OpTrait] = field(default_factory=frozenset)

    attribute_accessor_names: dict[str, str] = field(default_factory=dict)
    """
    Mapping from the accessor name to the attribute name.
    In some cases, the attribute name is not a valid Python identifier,
    or is already used by the operation, so we need to use a different name.
    """

    @staticmethod
    def from_pyrdl(pyrdl_def: type[_OpT]) -> OpDef:
        """Decorator used on classes to define a new operation definition."""

        type_var_mapping: dict[TypeVar, AttrConstraint] | None = None

        # If the operation inherit from `Generic`, this means that it specializes a
        # generic operation. Retrieve the mapping from `TypeVar` to pyrdl constraints.
        if issubclass(pyrdl_def, Generic):
            type_var_mapping = {
                k: irdl_to_attr_constraint(v)
                for k, v in get_type_var_mapping(pyrdl_def)[1].items()
            }

        def wrong_field_exception(field_name: str) -> PyRDLOpDefinitionError:
            raise PyRDLOpDefinitionError(
                f"{pyrdl_def.__name__}.{field_name} is neither a function, or an "
                "operand, result, region, or attribute definition. "
                "Operands should be defined with type hints of "
                "operand_def(<Constraint>), results with "
                "result_def(<Constraint>), regions with "
                "region_def(), and attributes with "
                "attr_def(<Constraint>)"
            )

        op_def = OpDef(pyrdl_def.name)

        # If an operation subclass overrides a superclass field, only keep the definition
        # of the subclass.
        field_names = set[str]()

        # Get all fields of the class, including the parent classes
        for parent_cls in pyrdl_def.mro():
            # Do not collect fields from Generic, as Generic will not contain
            # IRDL definitions, and contains ClassVar fields that are not
            # allowed in IRDL definitions.
            if parent_cls == Generic:
                continue
            if parent_cls in Operation.mro():
                continue

            clsdict = parent_cls.__dict__

            annotations = parent_cls.__annotations__

            for field_name in annotations:
                if field_name not in clsdict:
                    raise wrong_field_exception(field_name)

            for field_name in clsdict:
                if field_name == "name":
                    continue
                if field_name in _OPERATION_DICT_KEYS:
                    # Fields that are already in Operation (i.e. operands, results, ...)
                    continue
                if field_name in field_names:
                    # already registered value for field name
                    continue

                value = clsdict[field_name]

                # Check that all fields of the operation definition are either already
                # in Operation, or are class functions or methods.

                if field_name == "irdl_options":
                    if not isinstance(value, list):
                        assert False
                    op_def.options.extend(cast(list[Any], value))
                    continue

                if field_name == "traits":
                    traits = value
                    if not isinstance(traits, frozenset):
                        raise Exception(
                            f"pyrdl operation definition '{pyrdl_def.__name__}' "
                            f"has a 'traits' field of type {type(traits)}, but "
                            "it should be of type frozenset."
                        )
                    op_def.traits = traits
                    # Only register subclass traits
                    field_names.add(field_name)
                    continue

                # Dunder fields are allowed (i.e. __orig_bases__, __annotations__, ...)
                # They are used by Python to store information about the class, so they
                # should not be considered as part of the operation definition.
                # Also, they can provide a possiblea escape hatch.
                if field_name[:2] == "__" and field_name[-2:] == "__":
                    continue

                # Methods, properties, and functions are allowed
                if isinstance(
                    value, (FunctionType, PropertyType, classmethod, staticmethod)
                ):
                    continue
                # Constraint variables are allowed
                if get_origin(value) is Annotated:
                    if any(isinstance(arg, ConstraintVar) for arg in get_args(value)):
                        continue

                # Get attribute constraints from a list of pyrdl constraints
                def get_constraint(
                    pyrdl_constr: AttrConstraint
                    | Attribute
                    | type[Attribute]
                    | TypeVar,
                ) -> AttrConstraint:
                    return _irdl_list_to_attr_constraint(
                        (pyrdl_constr,),
                        allow_type_var=True,
                        type_var_mapping=type_var_mapping,
                    )

                field_names.add(field_name)

                match value:
                    case _ResultFieldDef():
                        constraint = get_constraint(value.param)
                        result_def = value.cls(constraint)
                        op_def.results.append((field_name, result_def))
                        continue
                    case _OperandFieldDef():
                        constraint = get_constraint(value.param)
                        attribute_def = value.cls(constraint)
                        op_def.operands.append((field_name, attribute_def))
                        continue
                    case _AttributeFieldDef():
                        constraint = get_constraint(value.param)
                        attribute_def = value.cls(constraint, attr_name=value.attr_name)
                        attr_name = (
                            field_name
                            if attribute_def.attr_name is None
                            else attribute_def.attr_name
                        )
                        op_def.attributes[attr_name] = attribute_def
                        op_def.attribute_accessor_names[field_name] = attr_name
                        continue
                    case _RegionFieldDef():
                        region_def = value.cls()
                        op_def.regions.append((field_name, region_def))
                        continue
                    case _SuccessorFieldDef():
                        successor_def = value.cls()
                        op_def.successors.append((field_name, successor_def))
                        continue
                    case _:
                        pass

                raise wrong_field_exception(field_name)

        return op_def

    def verify(self, op: Operation):
        """Given an IRDL definition, verify that an operation satisfies its invariants."""

        # Mapping from type variables to their concrete types.
        constraint_vars: dict[str, Attribute] = {}

        # Verify operands.
        irdl_op_verify_arg_list(op, self, VarIRConstruct.OPERAND, constraint_vars)

        # Verify results.
        irdl_op_verify_arg_list(op, self, VarIRConstruct.RESULT, constraint_vars)

        # Verify regions.
        irdl_op_verify_arg_list(op, self, VarIRConstruct.REGION, constraint_vars)

        # Verify successors.
        irdl_op_verify_arg_list(op, self, VarIRConstruct.SUCCESSOR, constraint_vars)

        # Verify attributes.
        for attr_name, attr_def in self.attributes.items():
            if attr_name not in op.attributes:
                if isinstance(attr_def, OptAttributeDef):
                    continue
                raise VerifyException(f"attribute {attr_name} expected")
            attr_def.constr.verify(op.attributes[attr_name], constraint_vars)

        # Verify traits.
        for trait in self.traits:
            trait.verify(op)


class VarIRConstruct(Enum):
    """
    An enum representing the part of an IR that may be variadic.
    This contains operands, results, and regions.
    """

    OPERAND = 1
    RESULT = 2
    REGION = 3
    SUCCESSOR = 4


def get_construct_name(construct: VarIRConstruct) -> str:
    """Get the type name, this is used mostly for error messages."""
    match construct:
        case VarIRConstruct.OPERAND:
            return "operand"
        case VarIRConstruct.RESULT:
            return "result"
        case VarIRConstruct.REGION:
            return "region"
        case VarIRConstruct.SUCCESSOR:
            return "successor"


def get_construct_defs(
    op_def: OpDef, construct: VarIRConstruct
) -> (
    list[tuple[str, OperandDef]]
    | list[tuple[str, ResultDef]]
    | list[tuple[str, RegionDef]]
    | list[tuple[str, SuccessorDef]]
):
    """Get the definitions of this type in an operation definition."""
    if construct == VarIRConstruct.OPERAND:
        return op_def.operands
    if construct == VarIRConstruct.RESULT:
        return op_def.results
    if construct == VarIRConstruct.REGION:
        return op_def.regions
    if construct == VarIRConstruct.SUCCESSOR:
        return op_def.successors
    assert False, "Unknown VarIRConstruct value"


def get_op_constructs(
    op: Operation, construct: VarIRConstruct
) -> Sequence[SSAValue] | list[OpResult] | list[Region] | list[Successor]:
    """
    Get the list of arguments of the type in an operation.
    For example, if the argument type is an operand, get the list of
    operands.
    """
    if construct == VarIRConstruct.OPERAND:
        return op.operands
    if construct == VarIRConstruct.RESULT:
        return op.results
    if construct == VarIRConstruct.REGION:
        return op.regions
    if construct == VarIRConstruct.SUCCESSOR:
        return op.successors
    assert False, "Unknown VarIRConstruct value"


def get_attr_size_option(
    construct: VarIRConstruct,
) -> (
    AttrSizedOperandSegments
    | AttrSizedResultSegments
    | AttrSizedRegionSegments
    | AttrSizedSuccessorSegments
):
    """Get the AttrSized option for this type."""
    if construct == VarIRConstruct.OPERAND:
        return AttrSizedOperandSegments()
    if construct == VarIRConstruct.RESULT:
        return AttrSizedResultSegments()
    if construct == VarIRConstruct.REGION:
        return AttrSizedRegionSegments()
    if construct == VarIRConstruct.SUCCESSOR:
        return AttrSizedSuccessorSegments()
    assert False, "Unknown VarIRConstruct value"


def get_variadic_sizes_from_attr(
    op: Operation,
    defs: Sequence[tuple[str, OperandDef | ResultDef | RegionDef | SuccessorDef]],
    construct: VarIRConstruct,
    size_attribute_name: str,
) -> list[int]:
    """
    Get the sizes of the variadic definitions
    from the corresponding attribute.
    """
    # Circular import because DenseArrayBase is defined using IRDL
    from xdsl.dialects.builtin import DenseArrayBase, i32

    # Check that the attribute is present
    if size_attribute_name not in op.attributes:
        raise VerifyException(
            f"Expected {size_attribute_name} attribute in {op.name} operation."
        )
    attribute = op.attributes[size_attribute_name]
    if not isinstance(attribute, DenseArrayBase):
        raise VerifyException(
            f"{size_attribute_name} attribute is expected " "to be a DenseArrayBase."
        )

    if attribute.elt_type != i32:
        raise VerifyException(
            f"{size_attribute_name} attribute is expected to "
            "be a DenseArrayBase of i32"
        )
    def_sizes = cast(list[int], [size_attr.data for size_attr in attribute.data.data])

    if len(def_sizes) != len(defs):
        raise VerifyException(
            f"expected {len(defs)} values in "
            f"{size_attribute_name}, but got {len(def_sizes)}"
        )

    variadic_sizes = list[int]()
    for (arg_name, arg_def), arg_size in zip(defs, def_sizes):
        if isinstance(arg_def, OptionalDef) and arg_size > 1:
            raise VerifyException(
                f"optional {get_construct_name(construct)} {arg_name} is expected to "
                f"be of size 0 or 1 in {size_attribute_name}, but got "
                f"{arg_size}"
            )

        if not isinstance(arg_def, VariadicDef) and arg_size != 1:
            raise VerifyException(
                f"non-variadic {get_construct_name(construct)} {arg_name} is expected "
                f"to be of size 1 in {size_attribute_name}, but got {arg_size}"
            )

        if isinstance(arg_def, VariadicDef):
            variadic_sizes.append(arg_size)

    return variadic_sizes


def get_variadic_sizes(
    op: Operation, op_def: OpDef, construct: VarIRConstruct
) -> list[int]:
    """Get variadic sizes of operands or results."""

    defs = get_construct_defs(op_def, construct)
    args = get_op_constructs(op, construct)
    def_type_name = get_construct_name(construct)
    attribute_option = get_attr_size_option(construct)

    variadic_defs = [
        (arg_name, arg_def)
        for arg_name, arg_def in defs
        if isinstance(arg_def, VariadicDef)
    ]

    # If the size is in the attributes, fetch it
    if attribute_option in op_def.options:
        return get_variadic_sizes_from_attr(
            op, defs, construct, attribute_option.attribute_name
        )

    # If there are no variadics arguments,
    # we just check that we have the right number of arguments
    if len(variadic_defs) == 0:
        if len(args) != len(defs):
            raise VerifyException(
                f"Expected {len(defs)} {def_type_name}, but got {len(args)}"
            )
        return []

    # If there is a single variadic argument,
    # we can get its size from the number of arguments.
    if len(variadic_defs) == 1:
        if len(args) - len(defs) + 1 < 0:
            raise VerifyException(
                f"Expected at least {len(defs) - 1} "
                f"{def_type_name}s, got {len(defs)}"
            )
        return [len(args) - len(defs) + 1]

    # Unreachable, all cases should have been handled.
    # Additional cases should raise an exception upon
    # definition of the irdl operation.
    assert False, "Unexpected xDSL error while fetching variadic sizes"


def get_operand_result_or_region(
    op: Operation,
    op_def: OpDef,
    arg_def_idx: int,
    previous_var_args: int,
    construct: VarIRConstruct,
) -> (
    None
    | SSAValue
    | Sequence[SSAValue]
    | list[OpResult]
    | Region
    | list[Region]
    | Successor
    | list[Successor]
):
    """
    Get an operand, result, or region.
    In the case of a variadic definition, return a list of elements.
    :param op: The operation we want to get argument of.
    :param arg_def_idx: The index of the argument in the irdl definition.
    :param previous_var_args: The number of previous variadic definitions
           before this definition.
    :param arg_type: The type of the argument we want
           (i.e. operand, result, or region)
    :return:
    """
    defs = get_construct_defs(op_def, construct)
    args = get_op_constructs(op, construct)

    variadic_sizes = get_variadic_sizes(op, op_def, construct)

    begin_arg = (
        arg_def_idx - previous_var_args + sum(variadic_sizes[:previous_var_args])
    )
    if isinstance(defs[arg_def_idx][1], OptionalDef):
        arg_size = variadic_sizes[previous_var_args]
        if arg_size == 0:
            return None
        else:
            return args[begin_arg]
    if isinstance(defs[arg_def_idx][1], VariadicDef):
        arg_size = variadic_sizes[previous_var_args]
        return args[begin_arg : begin_arg + arg_size]
    else:
        return args[begin_arg]


def irdl_op_verify_arg_list(
    op: Operation,
    op_def: OpDef,
    construct: VarIRConstruct,
    constraint_vars: dict[str, Attribute],
) -> None:
    """Verify the argument list of an operation."""
    arg_sizes = get_variadic_sizes(op, op_def, construct)
    arg_idx = 0
    var_idx = 0
    args = get_op_constructs(op, construct)

    def verify_arg(arg: Any, arg_def: Any, arg_idx: int) -> None:
        """Verify a single argument."""
        try:
            if (
                construct == VarIRConstruct.OPERAND
                or construct == VarIRConstruct.RESULT
            ):
                arg_def.constr.verify(arg.type, constraint_vars)
            elif construct == VarIRConstruct.REGION:
                if isinstance(arg_def, SingleBlockRegionDef) and len(arg.blocks) != 1:
                    raise VerifyException(
                        "expected a single block, but got " f"{len(arg.blocks)} blocks"
                    )
            elif construct == VarIRConstruct.SUCCESSOR:
                pass
            else:
                assert False, "Unknown VarIRConstruct value"
        except Exception as e:
            error(
                op,
                f"{get_construct_name(construct)} at position "
                f"{arg_idx} does not verify!\n{e}",
                e,
            )

    for def_idx, (_, arg_def) in enumerate(get_construct_defs(op_def, construct)):
        if isinstance(arg_def, VariadicDef):
            for _ in range(arg_sizes[var_idx]):
                verify_arg(args[arg_idx], arg_def, def_idx)
                arg_idx += 1
            var_idx += 1
        else:
            verify_arg(args[arg_idx], arg_def, def_idx)
            arg_idx += 1


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.OPERAND],
    args: Sequence[SSAValue | Sequence[SSAValue] | None],
    arg_defs: Sequence[tuple[str, OperandDef]],
    error_prefix: str,
) -> tuple[list[SSAValue], list[int]]:
    ...


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.RESULT],
    args: Sequence[Attribute | Sequence[Attribute] | None],
    arg_defs: Sequence[tuple[str, ResultDef]],
    error_prefix: str,
) -> tuple[list[Attribute], list[int]]:
    ...


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.REGION],
    args: Sequence[Region | Sequence[Region] | None],
    arg_defs: Sequence[tuple[str, RegionDef]],
    error_prefix: str,
) -> tuple[list[Region], list[int]]:
    ...


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.SUCCESSOR],
    args: Sequence[Successor | Sequence[Successor] | None],
    arg_defs: Sequence[tuple[str, SuccessorDef]],
    error_prefix: str,
) -> tuple[list[Successor], list[int]]:
    ...


_T = TypeVar("_T")


def irdl_build_arg_list(
    construct: VarIRConstruct,
    args: Sequence[_T | Sequence[_T] | None],
    arg_defs: Sequence[tuple[str, Any]],
    error_prefix: str = "",
) -> tuple[list[_T], list[int]]:
    """Build a list of arguments (operands, results, regions)"""

    if len(args) != len(arg_defs):
        raise ValueError(
            f"Expected {len(arg_defs)} {get_construct_name(construct)}, "
            f"but got {len(args)}"
        )

    res = list[_T]()
    arg_sizes = list[int]()

    for arg_idx, ((arg_name, arg_def), arg) in enumerate(zip(arg_defs, args)):
        if arg is None:
            if not isinstance(arg_def, OptionalDef):
                raise ValueError(
                    error_prefix
                    + f"passed None to a non-optional {construct} {arg_idx} '{arg_name}'"
                )
        elif isinstance(arg, Sequence):
            if not isinstance(arg_def, VariadicDef):
                raise ValueError(
                    error_prefix
                    + f"passed Sequence to non-variadic {construct} {arg_idx} '{arg_name}'"
                )
            arg = cast(Sequence[_T], arg)

            # Check we have at most one argument for optional defintions.
            if isinstance(arg_def, OptionalDef) and len(arg) > 1:
                raise ValueError(
                    error_prefix + f"optional {construct} {arg_idx} '{arg_name}' "
                    "expects a list of size at most 1 or None, but "
                    f"got a list of size {len(arg)}"
                )

            res.extend(arg)
            arg_sizes.append(len(arg))
        else:
            res.append(arg)
            arg_sizes.append(1)
    return res, arg_sizes


_OperandArg: TypeAlias = SSAValue | Operation


def irdl_build_operations_arg(
    operand: _OperandArg | Sequence[_OperandArg] | None,
) -> SSAValue | list[SSAValue]:
    if operand is None:
        return []
    elif isinstance(operand, SSAValue):
        return operand
    elif isinstance(operand, Operation):
        return SSAValue.get(operand)
    else:
        return [SSAValue.get(op) for op in operand]


_RegionArg: TypeAlias = Region | Sequence[Operation] | Sequence[Block]


def irdl_build_region_arg(r: _RegionArg) -> Region:
    if isinstance(r, Region):
        return r

    if not len(r):
        return Region()

    if isinstance(r[0], Operation):
        ops = cast(Sequence[Operation], r)
        return Region(Block(ops))
    else:
        return Region(cast(Sequence[Block], r))


def irdl_build_regions_arg(
    r: _RegionArg | Sequence[_RegionArg] | None,
) -> Region | list[Region]:
    if r is None:
        return []
    elif isinstance(r, Region):
        return r
    elif not len(r):
        return []
    elif isinstance(r[0], Operation):
        ops = cast(Sequence[Operation], r)
        return Region(Block(ops))
    elif isinstance(r[0], Block):
        blocks = cast(Sequence[Block], r)
        return Region(blocks)
    else:
        return [irdl_build_region_arg(_r) for _r in cast(Sequence[_RegionArg], r)]


def irdl_op_init(
    self: IRDLOperation,
    op_def: OpDef,
    operands: Sequence[SSAValue | Operation | Sequence[SSAValue | Operation] | None],
    res_types: Sequence[Attribute | Sequence[Attribute] | None],
    attributes: Mapping[str, Attribute | None],
    successors: Sequence[Successor | Sequence[Successor] | None],
    regions: Sequence[
        Region
        | Sequence[Operation]
        | Sequence[Block]
        | Sequence[Region | Sequence[Operation] | Sequence[Block]]
        | None
    ],
):
    """Builder for an irdl operation."""

    # We need irdl to define DenseArrayBase, but here we need
    # DenseArrayBase.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import DenseArrayBase, i32

    error_prefix = f"Error in {op_def.name} builder: "

    operands_arg = [irdl_build_operations_arg(operand) for operand in operands]

    regions_arg = [irdl_build_regions_arg(region) for region in regions]

    # Build the operands
    built_operands, operand_sizes = irdl_build_arg_list(
        VarIRConstruct.OPERAND, operands_arg, op_def.operands, error_prefix
    )

    # Build the results
    built_res_types, result_sizes = irdl_build_arg_list(
        VarIRConstruct.RESULT, res_types, op_def.results, error_prefix
    )

    # Build the regions
    built_regions, region_sizes = irdl_build_arg_list(
        VarIRConstruct.REGION, regions_arg, op_def.regions, error_prefix
    )

    # Build the successors
    built_successors, successor_sizes = irdl_build_arg_list(
        VarIRConstruct.SUCCESSOR, successors, op_def.successors, error_prefix
    )

    built_attributes = dict[str, Attribute]()
    for attr_name, attr in attributes.items():
        if attr is None:
            continue
        built_attributes[attr_name] = attr

    # Take care of variadic operand and result segment sizes.
    if AttrSizedOperandSegments() in op_def.options:
        built_attributes[
            AttrSizedOperandSegments.attribute_name
        ] = DenseArrayBase.from_list(i32, operand_sizes)

    if AttrSizedResultSegments() in op_def.options:
        built_attributes[
            AttrSizedResultSegments.attribute_name
        ] = DenseArrayBase.from_list(i32, result_sizes)

    if AttrSizedRegionSegments() in op_def.options:
        built_attributes[
            AttrSizedRegionSegments.attribute_name
        ] = DenseArrayBase.from_list(i32, region_sizes)

    if AttrSizedSuccessorSegments() in op_def.options:
        built_attributes[
            AttrSizedSuccessorSegments.attribute_name
        ] = DenseArrayBase.from_list(i32, successor_sizes)

    Operation.__init__(
        self,
        operands=built_operands,
        result_types=built_res_types,
        attributes=built_attributes,
        successors=built_successors,
        regions=built_regions,
    )


def irdl_op_arg_definition(
    new_attrs: dict[str, Any], construct: VarIRConstruct, op_def: OpDef
) -> None:
    previous_variadics = 0
    defs = get_construct_defs(op_def, construct)
    for arg_idx, (arg_name, arg_def) in enumerate(defs):

        def fun(self: Any, idx: int = arg_idx, previous_vars: int = previous_variadics):
            return get_operand_result_or_region(
                self, op_def, idx, previous_vars, construct
            )

        new_attrs[arg_name] = property(fun)
        if isinstance(arg_def, VariadicDef):
            previous_variadics += 1

    # If we have multiple variadics, check that we have an
    # attribute that holds the variadic sizes.
    arg_size_option = get_attr_size_option(construct)
    if previous_variadics > 1 and (arg_size_option not in op_def.options):
        arg_size_option_name = type(arg_size_option).__name__  # type: ignore
        raise Exception(
            f"Operation {op_def.name} defines more than two variadic "
            f"{get_construct_name(construct)}s, but do not define the "
            f"{arg_size_option_name} PyRDL option."
        )


def irdl_op_definition(cls: type[_OpT]) -> type[_OpT]:
    """Decorator used on classes to define a new operation definition."""

    assert issubclass(
        cls, IRDLOperation
    ), f"class {cls.__name__} should be a subclass of IRDLOperation"

    op_def = OpDef.from_pyrdl(cls)
    new_attrs = dict[str, Any]()

    # Add operand access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.OPERAND, op_def)

    # Add result access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.RESULT, op_def)

    # Add region access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.REGION, op_def)

    # Add successor access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.SUCCESSOR, op_def)

    def optional_attribute_field(attribute_name: str):
        def field_getter(self: _OpT):
            return self.attributes.get(attribute_name, None)

        def field_setter(self: _OpT, value: Attribute | None):
            if value is None:
                self.attributes.pop(attribute_name, None)
            else:
                self.attributes[attribute_name] = value

        return property(field_getter, field_setter)

    def attribute_field(attribute_name: str):
        def field_getter(self: _OpT):
            return self.attributes[attribute_name]

        def field_setter(self: _OpT, value: Attribute):
            self.attributes[attribute_name] = value

        return property(field_getter, field_setter)

    for accessor_name, attribute_name in op_def.attribute_accessor_names.items():
        attr_def = op_def.attributes[attribute_name]
        if isinstance(attr_def, OptAttributeDef):
            new_attrs[accessor_name] = optional_attribute_field(attribute_name)
        else:
            new_attrs[accessor_name] = attribute_field(attribute_name)

    new_attrs["traits"] = op_def.traits

    @classmethod
    @property
    def irdl_definition(cls: type[_OpT]):
        return op_def

    new_attrs["irdl_definition"] = irdl_definition

    custom_verify = getattr(cls, "verify_")

    def verify_(self: _OpT):
        op_def.verify(self)
        custom_verify(self)

    new_attrs["verify_"] = verify_

    return type(cls.__name__, cls.__mro__, {**cls.__dict__, **new_attrs})  # type: ignore


#  ____        _
# |  _ \  __ _| |_ __ _
# | | | |/ _` | __/ _` |
# | |_| | (_| | || (_| |
# |____/ \__,_|\__\__,_|
#

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

ParameterDef: TypeAlias = Annotated[_A, IRDLAnnotations.ParamDefAnnot]


def irdl_param_attr_get_param_type_hints(cls: type[_PAttrT]) -> list[tuple[str, Any]]:
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
    def from_pyrdl(pyrdl_def: type[ParametrizedAttribute]) -> ParamAttrDef:
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
                value, (FunctionType, PropertyType, classmethod, staticmethod)
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

        constraint_vars: dict[str, Attribute] = {}
        for param, (_, param_def) in zip(attr.parameters, self.parameters):
            param_def.verify(param, constraint_vars)


_PAttrT = TypeVar("_PAttrT", bound=ParametrizedAttribute)


def irdl_param_attr_definition(cls: type[_PAttrT]) -> type[_PAttrT]:
    """Decorator used on classes to define a new attribute definition."""

    attr_def = ParamAttrDef.from_pyrdl(cls)

    # New fields and methods added to the attribute
    new_fields = dict[str, Any]()

    def param_name_field(idx: int):
        @property
        def field(self: _PAttrT):
            return self.parameters[idx]

        return field

    for idx, (param_name, _) in enumerate(attr_def.parameters):
        new_fields[param_name] = param_name_field(idx)

    @classmethod
    @property
    def irdl_definition(cls: type[_PAttrT]):
        return attr_def

    new_fields["irdl_definition"] = irdl_definition

    return dataclass(frozen=True, init=False)(
        type(cls.__name__, (cls,), {**cls.__dict__, **new_fields})
    )  # type: ignore


_AttrT = TypeVar("_AttrT", bound=Attribute)


def irdl_attr_definition(cls: type[_AttrT]) -> type[_AttrT]:
    if issubclass(cls, ParametrizedAttribute):
        return irdl_param_attr_definition(cls)
    if issubclass(cls, Data):
        return dataclass(frozen=True)(  # pyright: ignore[reportGeneralTypeIssues]
            type(
                cls.__name__,
                (cls,),  # pyright: ignore[reportUnknownArgumentType]
                dict(cls.__dict__),
            )
        )
    raise Exception(
        f"Class {cls.__name__} should either be a subclass of 'Data' or "
        "'ParametrizedAttribute'"
    )
