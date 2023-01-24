from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from frozenlist import FrozenList
from functools import reduce
from inspect import isclass
from typing import (Annotated, Any, Callable, Generic, Sequence, TypeAlias,
                    TypeVar, Union, cast, get_args, get_origin, get_type_hints)
from types import UnionType, GenericAlias

from xdsl.ir import (Attribute, Block, Data, OpResult, Operation,
                     ParametrizedAttribute, Region, SSAValue)
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.exceptions import BuilderNotFoundException, VerifyException
from xdsl.utils.hints import is_satisfying_hint

# pyright: reportMissingParameterType=false, reportUnknownParameterType=false


def error(op: Operation, msg: str):
    diag = Diagnostic()
    diag.add_message(op, msg)
    diag.raise_exception(f"{op.name} operation does not verify", op)


class IRDLAnnotations(Enum):
    ParamDefAnnot = 1
    AttributeDefAnnot = 2
    OptAttributeDefAnnot = 3
    SingleBlockRegionAnnot = 4


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
    def verify(self, attr: Attribute) -> None:
        """
        Check if the attribute satisfies the constraint,
        or raise an exception otherwise.
        """
        ...


@dataclass
class EqAttrConstraint(AttrConstraint):
    """Constrain an attribute to be equal to another attribute."""

    attr: Attribute
    """The attribute we want to check equality with."""

    def verify(self, attr: Attribute) -> None:
        if attr != self.attr:
            raise VerifyException(
                f"Expected attribute {self.attr} but got {attr}")


@dataclass
class BaseAttr(AttrConstraint):
    """Constrain an attribute to be of a given base type."""

    attr: type[Attribute]
    """The expected attribute base type."""

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, self.attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.attr.name}")


def attr_constr_coercion(attr: (Attribute | type[Attribute]
                                | AttrConstraint)) -> AttrConstraint:
    """
    Attributes are coerced into EqAttrConstraints,
    and Attribute types are coerced into BaseAttr.
    """
    if isinstance(attr, Attribute):
        return EqAttrConstraint(attr)
    if isclass(attr) and issubclass(attr, Attribute):
        return BaseAttr(attr)
    assert (isinstance(attr, AttrConstraint))
    return attr


@dataclass
class AnyAttr(AttrConstraint):
    """Constraint that is verified by all attributes."""

    def verify(self, attr: Attribute) -> None:
        pass


@dataclass(init=False)
class AnyOf(AttrConstraint):
    """Ensure that an attribute satisfies one of the given constraints."""

    attr_constrs: list[AttrConstraint]
    """The list of constraints that are checked."""

    def __init__(self, attr_constrs: Sequence[Attribute | type[Attribute]
                                              | AttrConstraint]):
        self.attr_constrs = [
            attr_constr_coercion(constr) for constr in attr_constrs
        ]

    def verify(self, attr: Attribute) -> None:
        for attr_constr in self.attr_constrs:
            try:
                attr_constr.verify(attr)
                return
            except VerifyException:
                pass
        raise VerifyException(f"Unexpected attribute {attr}")


@dataclass()
class AllOf(AttrConstraint):
    """Ensure that an attribute satisfies all the given constraints."""

    attr_constrs: list[AttrConstraint]
    """The list of constraints that are checked."""

    def verify(self, attr: Attribute) -> None:
        for attr_constr in self.attr_constrs:
            attr_constr.verify(attr)


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

    def __init__(self, base_attr: type[ParametrizedAttribute],
                 param_constrs: Sequence[(Attribute | type[Attribute]
                                          | AttrConstraint)]):
        self.base_attr = base_attr
        self.param_constrs = [
            attr_constr_coercion(constr) for constr in param_constrs
        ]

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, self.base_attr):
            raise VerifyException(
                f"{attr} should be of base attribute {self.base_attr.name}")
        if len(self.param_constrs) != len(attr.parameters):
            raise VerifyException(
                f"{len(self.param_constrs)} parameters expected, "
                f"but got {len(attr.parameters)}")
        for idx, param_constr in enumerate(self.param_constrs):
            param_constr.verify(attr.parameters[idx])


def irdl_to_attr_constraint(
    irdl: Any,
    *,
    allow_type_var: bool = False,
    type_var_mapping: dict[TypeVar, AttrConstraint] | None = None
) -> AttrConstraint:
    if isinstance(irdl, AttrConstraint):
        return irdl

    if isinstance(irdl, Attribute):
        return EqAttrConstraint(irdl)

    # Annotated case
    # Each argument of the Annotated type corresponds to a constraint to satisfy.
    if get_origin(irdl) == Annotated:
        constraints: list[AttrConstraint] = []
        for arg in get_args(irdl):
            # We should not try to convert IRDL annotations, which do not
            # correspond to constraints
            if isinstance(arg, IRDLAnnotations):
                continue
            constraints.append(
                irdl_to_attr_constraint(arg,
                                        allow_type_var=allow_type_var,
                                        type_var_mapping=type_var_mapping))
        if len(constraints) > 1:
            return AllOf(constraints)
        return constraints[0]

    # Attribute class case
    # This is an `AnyAttr`, which does not constrain the attribute.
    if irdl is Attribute:
        return AnyAttr()

    # Attribute class case
    # This is a coercion for an `BaseAttr`.
    if isclass(irdl) and issubclass(irdl, Attribute):
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
            raise Exception("Type variables used in IRDL are expected to"
                            " be bound.")
        # We do not allow nested type variables.
        return irdl_to_attr_constraint(irdl.__bound__)

    origin = get_origin(irdl)

    # GenericData case
    if isclass(origin) and issubclass(origin, GenericData):
        return AllOf([
            BaseAttr(origin),
            origin.generic_constraint_coercion(get_args(irdl))
        ])

    # Generic ParametrizedAttributes case
    # We translate it to constraints over the attribute parameters.
    if isclass(origin) and issubclass(
            origin, ParametrizedAttribute) and issubclass(origin, Generic):
        args = [
            irdl_to_attr_constraint(arg,
                                    allow_type_var=allow_type_var,
                                    type_var_mapping=type_var_mapping)
            for arg in get_args(irdl)
        ]
        generic_args = ()

        # Get the Generic parent class to get the TypeVar parameters
        for parent in origin.__orig_bases__:  # type: ignore
            if get_origin(parent) == Generic:
                generic_args = get_args(parent)
                break
        else:
            raise Exception(
                f"Cannot parametrized non-generic {origin.name} attribute.")

        # Check that we have the right number of parameters
        if len(args) != len(generic_args):
            raise Exception(f"{origin.name} expects {len(generic_args)}"
                            f" parameters, got {len(args)}.")

        type_var_mapping = {
            parameter: arg
            for parameter, arg in zip(generic_args, args)
        }

        origin_parameters = irdl_param_attr_get_param_type_hints(origin)
        origin_constraints = [
            irdl_to_attr_constraint(param,
                                    allow_type_var=True,
                                    type_var_mapping=type_var_mapping)
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
                irdl_to_attr_constraint(arg,
                                        allow_type_var=allow_type_var,
                                        type_var_mapping=type_var_mapping))
        if len(constraints) > 1:
            return AnyOf(constraints)
        return constraints[0]

    # Better error messages for missing GenericData in Data definitions
    if isclass(origin) and issubclass(origin, Data):
        raise ValueError(
            f"Generic `Data` type '{origin.name}' cannot be converted to "
            "an attribute constraint. Consider making it inherit from "
            "`GenericData` instead of `Data`.")

    raise ValueError(f"Unexpected irdl constraint: {irdl}")


#   ___                       _   _
#  / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __
# | | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \
# | |_| | |_) |  __/ | | (_| | |_| | (_) | | | |
#  \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|
#       |_|

_OpT = TypeVar('_OpT', bound=Operation)


@dataclass
class IRDLOption(ABC):
    """Additional option used in IRDL."""
    ...


@dataclass
class AttrSizedOperandSegments(IRDLOption):
    """Expect an attribute on the op that contains the sizes of the variadic operands."""

    attribute_name = "operand_segment_sizes"
    """Name of the attribute containing the variadic operand sizes."""


@dataclass
class AttrSizedResultSegments(IRDLOption):
    """Expect an attribute on the op that contains the sizes of the variadic results."""

    attribute_name = "result_segment_sizes"
    """Name of the attribute containing the variadic result sizes."""


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

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(typ)


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

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(typ)


@dataclass(init=False)
class VarResultDef(ResultDef, VariadicDef):
    """An IRDL variadic result definition."""


VarOpResult: TypeAlias = list[OpResult]


@dataclass(init=False)
class OptResultDef(VarResultDef, OptionalDef):
    """An IRDL optional result definition."""


OptOpResult: TypeAlias = OpResult | None


@dataclass
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


SingleBlockRegion: TypeAlias = Annotated[
    Region, IRDLAnnotations.SingleBlockRegionAnnot]
VarSingleBlockRegion: TypeAlias = Annotated[
    list[Region], IRDLAnnotations.SingleBlockRegionAnnot]
OptSingleBlockRegion: TypeAlias = Annotated[
    Region | None, IRDLAnnotations.SingleBlockRegionAnnot]


@dataclass(init=False)
class AttributeDef:
    """An IRDL attribute definition."""

    constr: AttrConstraint
    """The attribute constraint."""

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(typ)


@dataclass(init=False)
class OptAttributeDef(AttributeDef):
    """An IRDL attribute definition for an optional attribute."""

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        super().__init__(typ)


_OpAttrT = TypeVar("_OpAttrT", bound=Attribute)

OpAttr: TypeAlias = Annotated[_OpAttrT, IRDLAnnotations.AttributeDefAnnot]
OptOpAttr: TypeAlias = Annotated[_OpAttrT | None,
                                 IRDLAnnotations.OptAttributeDefAnnot]


@dataclass(kw_only=True)
class OpDef:
    """The internal IRDL definition of an operation."""
    name: str = field(kw_only=False)
    operands: list[tuple[str, OperandDef]] = field(default_factory=list)
    results: list[tuple[str, ResultDef]] = field(default_factory=list)
    attributes: dict[str, AttributeDef] = field(default_factory=dict)
    regions: list[tuple[str, RegionDef]] = field(default_factory=list)
    options: list[IRDLOption] = field(default_factory=list)

    @staticmethod
    def from_pyrdl(pyrdl_def: type[_OpT]) -> OpDef:
        """Decorator used on classes to define a new operation definition."""

        # Get all fields of the class, including the parent classes
        clsdict: dict[str, Any] = dict()
        for parent_cls in pyrdl_def.mro()[::-1]:
            clsdict = {**clsdict, **parent_cls.__dict__}

        if "name" not in clsdict:
            raise Exception(
                f"pyrdl operation definition '{pyrdl_def.__name__}' does not "
                "define the operation name. The operation name is defined by "
                "adding a 'name' field.")

        op_def = OpDef(clsdict["name"])
        for field_name, field_type in get_type_hints(
                pyrdl_def, include_extras=True).items():

            if field_name in get_type_hints(Operation).keys():
                continue

            # If the field type is an Annotated, separate the origin
            # from the arguments.
            # If the field type is not an Annotated, then the arguments should
            # just be the field itself.
            origin = get_origin(field_type)
            args: tuple[Any, ...]
            if origin is None:
                args = (field_type, )
            elif origin == Annotated:
                args = get_args(field_type)
            else:
                args = (field_type, )
            args = cast(tuple[Any, ...], args)

            # Get attribute constraints from a list of pyrdl constraints
            def get_constraint(
                    pyrdl_constrs: tuple[Any, ...]) -> AttrConstraint:
                constraints = [
                    irdl_to_attr_constraint(pyrdl_constr)
                    for pyrdl_constr in pyrdl_constrs
                    if not isinstance(pyrdl_constr, IRDLAnnotations)
                ]
                if len(constraints) == 0:
                    return AnyAttr()
                if len(constraints) == 1:
                    return constraints[0]
                return AllOf(constraints)

            # Get the operand, result, attribute, or region definition, from
            # the pyrdl description.

            # For operands and results, constrants are encoded as arguments of
            # an Annotated, where the origin is the definition type (operand,
            # optional result, etc...).
            # For Attributes, constraints are encoded in the origin and the
            # args of the Annotated, and the definition type (required or
            # optional) is given in the Annotated arguments.
            # For Regions, SingleBlock regions are given as Annotated arguments,
            # and otherwise the Annotated origin (if it is an Annotated) gives
            # the Region definition (required, optional, or variadic).

            # Operand annotation
            if args[0] == Operand:
                constraint = get_constraint(args[1:])
                op_def.operands.append((field_name, OperandDef(constraint)))
            elif args[0] == list[Operand]:
                constraint = get_constraint(args[1:])
                op_def.operands.append((field_name, VarOperandDef(constraint)))
            elif args[0] == (Operand | None):
                constraint = get_constraint(args[1:])
                op_def.operands.append((field_name, OptOperandDef(constraint)))

            # Result annotation
            elif args[0] == OpResult:
                constraint = get_constraint(args[1:])
                op_def.results.append((field_name, ResultDef(constraint)))
            elif args[0] == list[OpResult]:
                constraint = get_constraint(args[1:])
                op_def.results.append((field_name, VarResultDef(constraint)))
            elif args[0] == (OpResult | None):
                constraint = get_constraint(args[1:])
                op_def.results.append((field_name, OptResultDef(constraint)))

            # Attribute annotation
            elif IRDLAnnotations.AttributeDefAnnot in args:
                constraint = get_constraint(args)
                op_def.attributes[field_name] = AttributeDef(constraint)
            elif IRDLAnnotations.OptAttributeDefAnnot in args:
                assert get_origin(args[0]) in [UnionType, Union]
                args = (reduce(lambda x, y: x | y,
                               get_args(args[0])[:-1]), *args[1:])
                constraint = get_constraint(args)
                op_def.attributes[field_name] = OptAttributeDef(constraint)

            # Region annotation
            elif args[0] == Region:
                if (len(args) > 1
                        and args[1] == IRDLAnnotations.SingleBlockRegionAnnot):
                    op_def.regions.append((field_name, SingleBlockRegionDef()))
                else:
                    op_def.regions.append((field_name, RegionDef()))
            elif args[0] == VarRegion:
                if (len(args) > 1
                        and args[1] == IRDLAnnotations.SingleBlockRegionAnnot):
                    op_def.regions.append(
                        (field_name, VarSingleBlockRegionDef()))
                else:
                    op_def.regions.append((field_name, VarRegionDef()))
            elif args[0] == OptRegion:
                if (len(args) > 1
                        and args[1] == IRDLAnnotations.SingleBlockRegionAnnot):
                    op_def.regions.append(
                        (field_name, OptSingleBlockRegionDef()))
                else:
                    op_def.regions.append((field_name, OptRegionDef()))

        op_def.options = clsdict.get("irdl_options", [])

        return op_def


class VarIRConstruct(Enum):
    """
    An enum representing the part of an IR that may be variadic.
    This contains operands, results, and regions.
    """
    OPERAND = 1
    RESULT = 2
    REGION = 3


def get_construct_name(construct: VarIRConstruct) -> str:
    """Get the type name, this is used mostly for error messages."""
    if construct == VarIRConstruct.OPERAND:
        return "operand"
    if construct == VarIRConstruct.RESULT:
        return "result"
    if construct == VarIRConstruct.REGION:
        return "region"
    assert False, "Unknown VarIRConstruct value"


def get_construct_defs(
    op_def: OpDef, construct: VarIRConstruct
) -> list[tuple[str, OperandDef]] | list[tuple[str, ResultDef]] | list[tuple[
        str, RegionDef]]:
    """Get the definitions of this type in an operation definition."""
    if construct == VarIRConstruct.OPERAND:
        return op_def.operands
    if construct == VarIRConstruct.RESULT:
        return op_def.results
    if construct == VarIRConstruct.REGION:
        return op_def.regions
    assert False, "Unknown VarIRConstruct value"


def get_op_constructs(
    op: Operation, construct: VarIRConstruct
) -> FrozenList[SSAValue] | list[OpResult] | list[Region]:
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
    assert False, "Unknown VarIRConstruct value"


def get_attr_size_option(
    construct: VarIRConstruct
) -> AttrSizedOperandSegments | AttrSizedResultSegments | None:
    """Get the AttrSized option for this type."""
    if construct == VarIRConstruct.OPERAND:
        return AttrSizedOperandSegments()
    if construct == VarIRConstruct.RESULT:
        return AttrSizedResultSegments()
    if construct == VarIRConstruct.REGION:
        return None
    assert False, "Unknown VarIRConstruct value"


def get_variadic_sizes_from_attr(op: Operation,
                                 defs: Sequence[tuple[str,
                                                      OperandDef | ResultDef]],
                                 construct: VarIRConstruct,
                                 size_attribute_name: str) -> list[int]:
    """
    Get the sizes of the variadic definitions
    from the corresponding attribute.
    """
    # Circular import because DenseIntOrFPElementsAttr is defined using IRDL
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    # Check that the attribute is present
    if size_attribute_name not in op.attributes:
        raise VerifyException(
            f"Expected {size_attribute_name} attribute in {op.name} operation."
        )
    attribute = op.attributes[size_attribute_name]
    if not isinstance(attribute, DenseIntOrFPElementsAttr):
        raise VerifyException(f"{size_attribute_name} attribute is expected "
                              "to be a DenseIntOrFPElementsAttr.")
    def_sizes: list[int] = [
        size_attr.value.data for size_attr in attribute.data.data
    ]
    if len(def_sizes) != len(defs):
        raise VerifyException(
            f"expected {len(defs)} values in "
            f"{size_attribute_name}, but got {len(def_sizes)}")

    variadic_sizes = list[int]()
    for ((arg_name, arg_def), arg_size) in zip(defs, def_sizes):
        if isinstance(arg_def, OptionalDef) and arg_size > 1:
            raise VerifyException(
                f"optional {get_construct_name(construct)} {arg_name} is expected to "
                f"be of size 0 or 1 in {size_attribute_name}, but got "
                f"{arg_size}")

        if not isinstance(arg_def, VariadicDef) and arg_size != 1:
            raise VerifyException(
                f"non-variadic {get_construct_name(construct)} {arg_name} is expected "
                f"to be of size 0 or 1 in {size_attribute_name}, but got "
                f"{arg_size}")

        if isinstance(arg_def, VariadicDef):
            variadic_sizes.append(arg_size)

    return variadic_sizes


def get_variadic_sizes(op: Operation, op_def: OpDef,
                       construct: VarIRConstruct) -> list[int]:
    """Get variadic sizes of operands or results."""

    defs = get_construct_defs(op_def, construct)
    args = get_op_constructs(op, construct)
    def_type_name = get_construct_name(construct)
    attribute_option = get_attr_size_option(construct)

    variadic_defs = [(arg_name, arg_def) for arg_name, arg_def in defs
                     if isinstance(arg_def, VariadicDef)]

    # If the size is in the attributes, fetch it
    if (attribute_option is not None) and (attribute_option in op_def.options):
        return get_variadic_sizes_from_attr(op, defs, construct,
                                            attribute_option.attribute_name)

    # If there are no variadics arguments,
    # we just check that we have the right number of arguments
    if len(variadic_defs) == 0:
        if len(args) != len(defs):
            raise VerifyException(
                f"Expected {len(defs)} {def_type_name}, but got {len(args)}")
        return []

    # If there is a single variadic argument,
    # we can get its size from the number of arguments.
    if len(variadic_defs) == 1:
        if len(args) - len(defs) + 1 < 0:
            raise VerifyException(f"Expected at least {len(defs) - 1} "
                                  f"{def_type_name}s, got {len(defs)}")
        return [len(args) - len(defs) + 1]

    # Unreachable, all cases should have been handled.
    # Additional cases should raise an exception upon
    # definition of the irdl operation.
    assert False, "Unexpected xDSL error while fetching variadic sizes"


def get_operand_result_or_region(
    op: Operation, op_def: OpDef, arg_def_idx: int, previous_var_args: int,
    construct: VarIRConstruct
) -> None | SSAValue | FrozenList[SSAValue] | list[OpResult] | Region | list[
        Region]:
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

    begin_arg = arg_def_idx - previous_var_args + sum(
        variadic_sizes[:previous_var_args])
    if isinstance(defs[arg_def_idx][1], OptionalDef):
        arg_size = variadic_sizes[previous_var_args]
        if arg_size == 0:
            return None
        else:
            return args[begin_arg]
    if isinstance(defs[arg_def_idx][1], VariadicDef):
        arg_size = variadic_sizes[previous_var_args]
        return args[begin_arg:begin_arg + arg_size]
    else:
        return args[begin_arg]


def irdl_op_verify_arg_list(op: Operation, op_def: OpDef,
                            construct: VarIRConstruct) -> None:
    """Verify the argument list of an operation."""
    arg_sizes = get_variadic_sizes(op, op_def, construct)
    arg_idx = 0
    var_idx = 0
    args = get_op_constructs(op, construct)

    def verify_arg(arg: Any, arg_def: Any, arg_idx: int) -> None:
        """Verify a single argument."""
        try:
            if construct == VarIRConstruct.OPERAND or construct == VarIRConstruct.RESULT:
                arg_def.constr.verify(arg.typ)
            elif construct == VarIRConstruct.REGION:
                if isinstance(arg_def,
                              SingleBlockRegionDef) and len(arg.blocks) != 1:
                    raise VerifyException("expected a single block, but got "
                                          f"{len(arg.blocks)} blocks")
            else:
                assert False, "Unknown ArgType value"
        except Exception as e:
            error(
                op, f"{get_construct_name(construct)} at position "
                f"{arg_idx} does not verify!\n{e}")

    for def_idx, (_,
                  arg_def) in enumerate(get_construct_defs(op_def, construct)):
        if isinstance(arg_def, VariadicDef):
            for _ in range(arg_sizes[var_idx]):
                verify_arg(args[arg_idx], arg_def, def_idx)
                arg_idx += 1
            var_idx += 1
        else:
            verify_arg(args[arg_idx], arg_def, def_idx)
            arg_idx += 1


def irdl_op_verify(op: Operation, op_def: OpDef) -> None:
    """Given an IRDL definition, verify that an operation satisfies its invariants."""

    # Verify operands.
    irdl_op_verify_arg_list(op, op_def, VarIRConstruct.OPERAND)

    # Verify results.
    irdl_op_verify_arg_list(op, op_def, VarIRConstruct.RESULT)

    # Verify regions.
    irdl_op_verify_arg_list(op, op_def, VarIRConstruct.REGION)

    for attr_name, attr_def in op_def.attributes.items():
        if attr_name not in op.attributes:
            if isinstance(attr_def, OptAttributeDef):
                continue
            raise VerifyException(f"attribute {attr_name} expected")
        attr_def.constr.verify(op.attributes[attr_name])


def irdl_build_attribute(irdl_def: AttrConstraint, result: Any) -> Attribute:
    if isinstance(irdl_def, BaseAttr):
        if isinstance(result, tuple):
            return irdl_def.attr.build(*result)
        return irdl_def.attr.build(result)
    if isinstance(result, Attribute):
        return result
    raise Exception(f"builder expected an attribute, got {result}")


def irdl_build_arg_list(construct: VarIRConstruct,
                        args: Sequence[Any],
                        arg_defs: Sequence[tuple[str, Any]],
                        error_prefix: str = "") -> tuple[list[Any], list[int]]:
    """Build a list of arguments (operands, results, regions)"""

    def build_arg(arg_def: Any, arg: Any) -> Any:
        """Build a single argument."""
        if construct == VarIRConstruct.OPERAND:
            return SSAValue.get(arg)
        elif construct == VarIRConstruct.RESULT:
            assert isinstance(arg_def, ResultDef)
            return irdl_build_attribute(arg_def.constr, arg)
        elif construct == VarIRConstruct.REGION:
            assert isinstance(arg_def, RegionDef)
            return Region.get(arg)
        else:
            assert False, "Unknown ArgType value"

    if len(args) != len(arg_defs):
        raise ValueError(
            f"Expected {len(arg_defs)} {get_construct_name(construct)}, "
            f"but got {len(args)}")

    res = list[Any]()
    arg_sizes = list[int]()

    for arg_idx, ((arg_name, arg_def), arg) in enumerate(zip(arg_defs, args)):
        if isinstance(arg_def, VariadicDef):
            if not isinstance(arg, list):
                raise ValueError(
                    error_prefix +
                    f"variadic {construct} {arg_idx} '{arg_name}' "
                    f" expects a list, but got {arg}")
            arg = cast(list[Any], arg)

            # Check we have at most one argument for optional defintions.
            if isinstance(arg_def, OptionalDef) and len(arg) > 1:
                raise ValueError(
                    error_prefix +
                    f"optional {construct} {arg_idx} '{arg_name}' "
                    "expects a list of size at most 1, but "
                    f"got a list of size {len(arg)}")

            res.extend([build_arg(arg_def, arg_arg) for arg_arg in arg])
            arg_sizes.append(len(arg))
        else:
            res.append(build_arg(arg_def, arg))
            arg_sizes.append(1)
    return res, arg_sizes


def irdl_op_builder(cls: type[_OpT], op_def: OpDef,
                    operands: Sequence[SSAValue | Operation
                                       | list[SSAValue | Operation] | None],
                    res_types: Sequence[Any | list[Any] | None],
                    attributes: dict[str, Any], successors: Sequence[Block],
                    regions: Sequence[Any | None]) -> _OpT:
    """Builder for an irdl operation."""

    # We need irdl to define DenseIntOrFPElementsAttr, but here we need
    # DenseIntOrFPElementsAttr.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, i32)

    error_prefix = f"Error in {op_def.name} builder: "

    # Build the operands
    built_operands, operand_sizes = irdl_build_arg_list(
        VarIRConstruct.OPERAND, operands, op_def.operands, error_prefix)

    # Build the results
    built_res_types, result_sizes = irdl_build_arg_list(
        VarIRConstruct.RESULT, res_types, op_def.results, error_prefix)

    # Build the regions
    built_regions, _ = irdl_build_arg_list(VarIRConstruct.REGION, regions,
                                           op_def.regions, error_prefix)

    # Build attributes by forwarding the values to the attribute builders
    attr_defs = {name: def_ for (name, def_) in op_def.attributes.items()}

    built_attributes = dict[str, Attribute]()
    for attr_name, attr in attributes.items():
        if attr_name not in attr_defs:
            if isinstance(attr, Attribute):
                built_attributes[attr_name] = attr
                continue
            raise ValueError(error_prefix +
                             f"unexpected attribute name {attr_name}.")
        built_attributes[attr_name] = irdl_build_attribute(
            attr_defs[attr_name].constr, attr)

    # Take care of variadic operand and result segment sizes.
    if AttrSizedOperandSegments() in op_def.options:
        sizes = operand_sizes
        built_attributes[AttrSizedOperandSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    if AttrSizedResultSegments() in op_def.options:
        sizes = result_sizes
        built_attributes[AttrSizedResultSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    return cls.create(operands=built_operands,
                      result_types=built_res_types,
                      attributes=built_attributes,
                      successors=successors,
                      regions=built_regions)


def irdl_op_arg_definition(new_attrs: dict[str, Any],
                           construct: VarIRConstruct, op_def: OpDef) -> None:
    previous_variadics = 0
    defs = get_construct_defs(op_def, construct)
    for arg_idx, (arg_name, arg_def) in enumerate(defs):

        def fun(self: Any,
                idx: int = arg_idx,
                previous_vars: int = previous_variadics):
            return get_operand_result_or_region(self, op_def, idx,
                                                previous_vars, construct)

        new_attrs[arg_name] = property(fun)
        if isinstance(arg_def, VariadicDef):
            previous_variadics += 1

    # If we have multiple variadics, check that we have an
    # attribute that holds the variadic sizes.
    arg_size_option = get_attr_size_option(construct)
    if previous_variadics > 1 and (arg_size_option is None
                                   or arg_size_option not in op_def.options):
        if arg_size_option is None:
            arg_size_option_name = 'unknown'
        else:
            arg_size_option_name = arg_size_option.__name__  # type: ignore
        raise Exception(
            "Operation defines more than two variadic "
            f"{get_construct_name(construct)}s, but do not define the "
            f"{arg_size_option_name} PyRDL option.")


def irdl_op_definition(cls: type[_OpT]) -> type[_OpT]:
    """Decorator used on classes to define a new operation definition."""

    assert issubclass(
        cls,
        Operation), f"class {cls.__name__} should be a subclass of Operation"

    # Get all fields of the class, including the parent classes
    clsdict = dict[str, Any]()
    for parent_cls in cls.mro()[::-1]:
        clsdict = {**clsdict, **parent_cls.__dict__}

    op_def = OpDef.from_pyrdl(cls)
    new_attrs = dict[str, Any]()

    # Add operand access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.OPERAND, op_def)

    # Add result access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.RESULT, op_def)

    # Add region access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.REGION, op_def)

    for attribute_name, attr_def in op_def.attributes.items():
        if isinstance(attr_def, OptAttributeDef):
            new_attrs[attribute_name] = property(
                lambda self, name=attribute_name: self.attributes.get(
                    name, None))
        else:
            new_attrs[attribute_name] = property(
                lambda self, name=attribute_name: self.attributes[name])

    new_attrs["verify_"] = lambda op: irdl_op_verify(op, op_def)
    if "verify_" in clsdict:
        custom_verifier = clsdict["verify_"]

        def new_verifier(verifier, op):
            verifier(op)
            custom_verifier(op)

        new_attrs["verify_"] = (
            lambda verifier: lambda op: new_verifier(verifier, op))(
                new_attrs["verify_"])

    def builder(cls,
                operands=[],
                result_types=[],
                attributes=dict(),
                successors=[],
                regions=[]):
        return irdl_op_builder(cls, op_def, operands, result_types, attributes,
                               successors, regions)

    new_attrs["build"] = classmethod(builder)
    new_attrs["irdl_definition"] = classmethod(property(lambda cls: op_def))

    return type(cls.__name__, cls.__mro__, {**cls.__dict__, **new_attrs})


#     _   _   _        _ _           _
#    / \ | |_| |_ _ __(_) |__  _   _| |_ ___
#   / _ \| __| __| '__| | '_ \| | | | __/ _ \
#  / ___ \ |_| |_| |  | | |_) | |_| | ||  __/
# /_/   \_\__|\__|_|  |_|_.__/ \__,_|\__\___|
#

_AttrT = TypeVar('_AttrT', bound=Attribute)

_BuilderTyT = TypeVar("_BuilderTyT", bound=Attribute)

BuilderTy: TypeAlias = Callable[..., _BuilderTyT]

IRDL_IS_BUILDER = '__irdl_is_builder'


def builder(f: BuilderTy[_AttrT]) -> BuilderTy[_AttrT]:
    """
    Annotate a function and mark it as an IRDL builder.
    This should only be used as decorator in classes decorated by irdl_attr_builder.
    """
    setattr(f, IRDL_IS_BUILDER, True)
    return f


def irdl_get_builders(cls: type[_AttrT]) -> list[BuilderTy[_AttrT]]:
    """Get functions decorated with 'builder' in a class."""
    builders = list[BuilderTy[_AttrT]]()
    for field_name in cls.__dict__:
        field_ = cls.__dict__[field_name]
        # Builders are staticmethods, so we need to get back the original function
        # with __func__
        if hasattr(field_, "__func__") and hasattr(field_.__func__,
                                                   IRDL_IS_BUILDER):
            builders.append(field_.__func__)
    return builders


#  ____        _
# |  _ \  __ _| |_ __ _
# | | | |/ _` | __/ _` |
# | |_| | (_| | || (_| |
# |____/ \__,_|\__\__,_|
#

_DataElement = TypeVar("_DataElement")


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


_DT = TypeVar("_DT")


def irdl_data_verify(data: Data[_DT], typ: type[_DT]) -> None:
    """Check that the Data has the expected type."""
    if isinstance(data.data, typ):
        return
    raise VerifyException(
        f"{data.name} data attribute expected type {typ}, but {type(data.data)} given."
    )


T = TypeVar('T', bound=Data[Any])


def irdl_data_definition(cls: type[T]) -> type[T]:
    """Decorator to transform an IRDL Data definition to a Python class."""
    new_attrs = dict[str, Any]()

    # Build method is added for all definitions.
    if "build" in cls.__dict__:
        raise Exception(
            f'"build" method for {cls.__name__} is reserved for IRDL, '
            f'and should not be defined.')
    builders = irdl_get_builders(cls)
    new_attrs["build"] = lambda *args: irdl_attr_builder(cls, builders, *args)

    # Verify method is added if not redefined by the user.
    if "verify" not in cls.__dict__:
        for parent in cls.__orig_bases__:
            if get_origin(parent) != Data:
                continue
            if len(get_args(parent)) != 1:
                raise Exception(f"In {cls.__name__} definition: Data expects "
                                "a single type parameter")
            expected_type = get_args(parent)[0]
            if not isclass(expected_type):
                raise Exception(f'In {cls.__name__} definition: Cannot infer '
                                f'"verify" method. Type parameter of Data is '
                                f'not a class.')
            if isinstance(expected_type, GenericAlias):
                raise Exception(f'In {cls.__name__} definition: Cannot infer '
                                f'"verify" method. Type parameter of Data has '
                                f'type GenericAlias.')
            new_attrs[
                "verify"] = lambda self, expected_type=expected_type: irdl_data_verify(
                    self, expected_type)
            break
        else:
            raise Exception(f'Missing method "verify" in {cls.__name__} data '
                            'attribute definition: the "verify" method cannot '
                            'be automatically derived for this definition.')

    return dataclass(frozen=True)(type(cls.__name__, (cls, ), {
        **cls.__dict__,
        **new_attrs
    }))


#  ____                              _   _   _
# |  _ \ __ _ _ __ __ _ _ __ ___    / \ | |_| |_ _ __
# | |_) / _` | '__/ _` | '_ ` _ \  / _ \| __| __| '__|
# |  __/ (_| | | | (_| | | | | | |/ ___ \ |_| |_| |
# |_|   \__,_|_|  \__,_|_| |_| |_/_/   \_\__|\__|_|
#

_A = TypeVar("_A", bound=Attribute)

ParameterDef: TypeAlias = Annotated[_A, IRDLAnnotations.ParamDefAnnot]


def irdl_param_attr_get_param_type_hints(
        cls: type[_PAttrT]) -> list[tuple[str, Any]]:
    """Get the type hints of an IRDL parameter definitions."""
    res = list[tuple[str, Any]]()
    for field_name, field_type in get_type_hints(cls,
                                                 include_extras=True).items():
        if field_name == "name" or field_name == "parameters":
            continue

        origin = get_origin(field_type)
        args = get_args(field_type)
        if origin != Annotated or IRDLAnnotations.ParamDefAnnot not in args:
            raise ValueError(
                f"In attribute {cls.__name__} definition: Parameter " +
                f"definition {field_name} should be defined with " +
                f"type `ParameterDef`, got type {field_type}.")

        res.append((field_name, field_type))
    return res


@dataclass
class ParamAttrDef:
    """The IRDL definition of a parametrized attribute."""
    name: str
    parameters: list[tuple[str, AttrConstraint]]

    @staticmethod
    def from_pyrdl(pyrdl_def: type[ParametrizedAttribute]) -> ParamAttrDef:
        # Get the fields from the class and its parents
        clsdict = dict[str, Any]()
        for parent_cls in pyrdl_def.mro()[::-1]:
            clsdict = {**clsdict, **parent_cls.__dict__}

        if "name" not in clsdict:
            raise Exception(
                f"pyrdl attribute definition '{pyrdl_def.__name__}' does not "
                "define the attribute name. The attribute name is defined by "
                "adding a 'name' field.")

        name = clsdict["name"]

        param_hints = irdl_param_attr_get_param_type_hints(pyrdl_def)

        parameters = list[tuple[str, AttrConstraint]]()
        for param_name, param_type in param_hints:
            constraint = irdl_to_attr_constraint(param_type,
                                                 allow_type_var=True)
            parameters.append((param_name, constraint))

        return ParamAttrDef(name, parameters)


def irdl_attr_verify(attr: ParametrizedAttribute, attr_def: ParamAttrDef):
    """Given an IRDL definition, verify that an attribute satisfies its invariants."""

    if len(attr.parameters) != len(attr_def.parameters):
        raise VerifyException(
            f"In {attr_def.name} attribute verifier: "
            f"{len(attr_def.parameters)} parameters expected, got "
            f"{len(attr.parameters)}")

    for param, (_, param_def) in zip(attr.parameters, attr_def.parameters):
        param_def.verify(param)


_PAttrT = TypeVar('_PAttrT', bound=ParametrizedAttribute)


def irdl_attr_try_builder(
        builder: BuilderTy[_PAttrT],
        *args: tuple[Any, ...]) -> ParametrizedAttribute | None:
    params_dict = get_type_hints(builder)
    builder_params = inspect.signature(builder).parameters
    params = [params_dict[param.name] for param in builder_params.values()]
    defaults = [param.default for param in builder_params.values()]
    num_non_defaults = defaults.count(inspect.Signature.empty)
    if num_non_defaults > len(args):
        return None
    if len(params) < len(args):
        return None
    for arg, param in zip(args, params[:len(args)]):
        if not is_satisfying_hint(arg, param):
            return None
    return builder(*args, *defaults[len(args):])


def irdl_attr_builder(cls: type[_PAttrT],
                      builders: Sequence[BuilderTy[_PAttrT]],
                      *args: tuple[Any, ...]):
    """Try to apply all builders to construct an attribute instance."""
    if len(args) == 1 and isinstance(args[0], cls):
        return args[0]
    for builder in builders:
        res = irdl_attr_try_builder(builder, *args)
        if res:
            return res
    raise BuilderNotFoundException(cls, args)


def irdl_param_attr_definition(cls: type[_PAttrT]) -> type[_PAttrT]:
    """Decorator used on classes to define a new attribute definition."""

    # Get the fields from the class and its parents
    clsdict = dict[str, Any]()
    for parent_cls in cls.mro()[::-1]:
        clsdict = {**clsdict, **parent_cls.__dict__}

    attr_def = ParamAttrDef.from_pyrdl(cls)

    # New fields and methods added to the attribute
    new_fields = dict[str, Any]()

    for idx, (param_name, _) in enumerate(attr_def.parameters):
        new_fields[param_name] = property(
            lambda self, idx=idx: self.parameters[idx])

    new_fields["verify"] = lambda typ: irdl_attr_verify(typ, attr_def)

    if "verify" in clsdict:
        custom_verifier = clsdict["verify"]

        def new_verifier(verifier, op):
            verifier(op)
            custom_verifier(op)

        new_fields["verify"] = (
            lambda verifier: lambda op: new_verifier(verifier, op))(
                new_fields["verify"])

    builders = irdl_get_builders(cls)
    if "build" in cls.__dict__:
        raise Exception(
            f'"build" method for {cls.__name__} is reserved for IRDL, ' +
            'and should not be defined.')
    new_fields["build"] = lambda *args: irdl_attr_builder(cls, builders, *args)

    new_fields["irdl_definition"] = classmethod(property(lambda cls: attr_def))

    return dataclass(frozen=True, init=False)(type(cls.__name__, (cls, ), {
        **cls.__dict__,
        **new_fields
    }))


def irdl_attr_definition(cls: type[_AttrT]) -> type[_AttrT]:
    if issubclass(cls, ParametrizedAttribute):
        return irdl_param_attr_definition(cls)
    if issubclass(cls, Data):
        return irdl_data_definition(cls)
    raise Exception(
        f"Class {cls.__name__} should either be a subclass of 'Data' or "
        "'ParametrizedAttribute'")
