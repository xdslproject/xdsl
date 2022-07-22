from __future__ import annotations

import inspect
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from inspect import isclass
from typing import (Annotated, Any, Callable, Generic, Sequence, TypeAlias,
                    TypeVar, Union, cast, get_args, get_origin, get_type_hints)

from frozenlist import FrozenList

from xdsl import util
from xdsl.diagnostic import Diagnostic, DiagnosticException
from xdsl.ir import (Attribute, Block, Data, OpResult, Operation,
                     ParametrizedAttribute, Region, SSAValue)


def error(op: Operation, msg: str):
    diag = Diagnostic()
    diag.add_message(op, msg)
    diag.raise_exception(f"{op.name} operation does not verify", op)


class VerifyException(DiagnosticException):
    ...


class IRDLAnnotations(Enum):
    ParamDefAnnot = 1


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
        if not isinstance(attr, Attribute):
            raise VerifyException(f"Expected attribute, but got {attr}")


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
            if isinstance(attr_constr, Attribute) and attr_constr == attr:
                return
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

    base_attr: type[Attribute]
    """The base attribute type."""

    param_constrs: list[AttrConstraint]
    """The attribute parameter constraints"""

    def __init__(self, base_attr: type[Attribute],
                 param_constrs: list[(Attribute | type[Attribute]
                                      | AttrConstraint)]):
        self.base_attr = base_attr
        self.param_constrs = [
            attr_constr_coercion(constr) for constr in param_constrs
        ]

    def verify(self, attr: Attribute) -> None:
        assert isinstance(attr, ParametrizedAttribute)
        if not isinstance(attr, self.base_attr):
            # the type checker concludes that attr has type 'Never', therefore the cast
            name = cast(Attribute, attr).name
            raise VerifyException(
                f"Base attribute {self.base_attr.name} expected, but got {name}"
            )
        if len(self.param_constrs) != len(attr.parameters):
            raise VerifyException(
                f"{len(self.param_constrs)} parameters expected, but got {len(attr.parameters)}"
            )
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

    # Annotated case
    # Each argument of the Annotated type correspond to a constraint to satisfy.
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
        if type_var_mapping is not None:
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
        origin_constraints: list[Attribute | type[Attribute]
                                 | AttrConstraint] = [
                                     irdl_to_attr_constraint(
                                         param,
                                         allow_type_var=True,
                                         type_var_mapping=type_var_mapping)
                                     for _, param in origin_parameters
                                 ]
        return ParamAttrConstraint(origin, origin_constraints)

    # Union case
    # This is a coercion for an `AnyOf` constraint.
    if origin == types.UnionType or origin == Union:
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
class OptionalDef(OperandOrResultDef):
    """An optional operand or result definition. Should not be used directly."""
    ...


@dataclass(init=False)
class OperandDef(OperandOrResultDef):
    """An IRDL operand definition."""

    constr: AttrConstraint
    """The operand constraint."""

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(typ)


@dataclass(init=False)
class VarOperandDef(OperandDef, VariadicDef):
    """An IRDL variadic operand definition."""


@dataclass(init=False)
class OptOperandDef(VarOperandDef, OptionalDef):
    """An IRDL optional operand definition."""


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


@dataclass(init=False)
class OptResultDef(VarResultDef, OptionalDef):
    """An IRDL optional result definition."""


@dataclass
class RegionDef(Region):
    """
    An IRDL region definition.
    If the block_args is specified, then the region expect to have the entry block with these arguments.
    """
    block_args: list[Attribute] | None = None
    blocks: list[Block] = field(default_factory=list)


@dataclass
class VarRegionDef(RegionDef, VariadicDef):
    """An IRDL variadic region definition."""


@dataclass
class OptRegionDef(RegionDef, OptionalDef):
    """An IRDL optional region definition."""


@dataclass
class SingleBlockRegionDef(RegionDef):
    """An IRDL region definition that expects exactly one block."""
    pass


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

        for field_name, field in clsdict.items():
            if isinstance(field, OperandDef):
                op_def.operands.append((field_name, field))
            elif isinstance(field, ResultDef):
                op_def.results.append((field_name, field))
            elif isinstance(field, RegionDef):
                op_def.regions.append((field_name, field))
            elif isinstance(field, AttributeDef):
                op_def.attributes[field_name] = field

        op_def.options = clsdict.get("irdl_options", [])

        return op_def


def get_variadic_sizes_from_attr(
        op: Operation, n_expected_variadics: int,
        option: AttrSizedOperandSegments | AttrSizedResultSegments
) -> list[int]:
    # Circular import because DenseIntOrFPElementsAttr is defined using IRDL
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    size_attribute_name = option.attribute_name

    if size_attribute_name not in op.attributes:
        raise VerifyException(
            f"Expected {size_attribute_name} attribute in {op.name} operation."
        )
    attribute = op.attributes[size_attribute_name]
    if not isinstance(attribute, DenseIntOrFPElementsAttr):
        raise VerifyException(
            f"{size_attribute_name} attribute is expected to be a DenseIntOrFPElementsAttr."
        )
    variadic_sizes: list[int] = [
        size_attr.value.data for size_attr in attribute.data.data
    ]
    if len(variadic_sizes) != n_expected_variadics:
        raise VerifyException(
            f"expected {n_expected_variadics} values in "
            f"{size_attribute_name}, but got {len(variadic_sizes)}")
    return variadic_sizes


class VariadicType(Enum):
    OPERAND = 1
    RESULT = 2
    REGION = 3

    def get_defs(
        self, op_def: OpDef
    ) -> list[tuple[str, OperandDef]] | list[tuple[str, ResultDef]] | list[
            tuple[str, RegionDef]]:
        if self == self.OPERAND:
            return op_def.operands
        if self == self.RESULT:
            return op_def.results
        if self == self.REGION:
            return op_def.regions
        assert False, "Unknown VariadicType value"

    def get_args(
            self, op: Operation
    ) -> FrozenList[SSAValue] | list[OpResult] | list[Region]:
        if self == self.OPERAND:
            return op.operands
        if self == self.RESULT:
            return op.results
        if self == self.REGION:
            return op.regions
        assert False, "Unknown VariadicType value"

    def get_name(self) -> str:
        if self == self.OPERAND:
            return "operand"
        if self == self.RESULT:
            return "result"
        if self == self.REGION:
            return "region"
        assert False, "Unknown VariadicType value"

    def get_attr_size_option(
            self) -> AttrSizedOperandSegments | AttrSizedResultSegments | None:
        if self == self.OPERAND:
            return AttrSizedOperandSegments()
        if self == self.RESULT:
            return AttrSizedResultSegments()
        if self == self.REGION:
            return None
        assert False, "Unknown VariadicType value"


def get_variadic_sizes(op: Operation, op_def: OpDef,
                       typ: VariadicType) -> list[int]:
    """Get variadic sizes of operands or results."""

    defs = typ.get_defs(op_def)
    args = typ.get_args(op)
    def_type_name = typ.get_name()
    attribute_option = typ.get_attr_size_option()

    variadic_defs = [(arg_name, arg_def) for arg_name, arg_def in defs
                     if isinstance(arg_def, VariadicDef)]

    # If the size is in the attributes, fetch it
    if (attribute_option is not None) and (attribute_option in op_def.options):
        return get_variadic_sizes_from_attr(op, len(variadic_defs),
                                            attribute_option)

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
    arg_type: VariadicType
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
    defs = arg_type.get_defs(op_def)
    args = arg_type.get_args(op)

    variadic_sizes = get_variadic_sizes(op, op_def, arg_type)

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
                            arg_type: VariadicType) -> None:
    arg_sizes = get_variadic_sizes(op, op_def, arg_type)
    arg_idx = 0
    def_idx = 0
    args = arg_type.get_args(op)

    def verify_arg(arg: Any, arg_def: Any, arg_idx: int) -> None:
        try:
            if arg_type == VariadicType.OPERAND or arg_type == VariadicType.RESULT:
                arg_def.constr.verify(arg.typ)
            elif arg_type == VariadicType.REGION:
                arg_def.constr.verify(arg)
            else:
                assert False, "Unknown VariadicType value"
        except Exception as e:
            error(
                op,
                f"{arg_type.get_name} at position {arg_idx} does not verify!\n{e}"
            )

    for def_idx, (_, arg_def) in enumerate(arg_type.get_defs(op_def)):
        if isinstance(arg_def, VariadicDef):
            for _ in range(arg_sizes[def_idx]):
                verify_arg(args[arg_idx], arg_def, def_idx)
                arg_idx += 1
        else:
            verify_arg(args[arg_idx], arg_def, def_idx)
            arg_idx += 1


def irdl_op_verify(op: Operation, op_def: OpDef) -> None:
    """Given an IRDL definition, verify that an operation satisfies its invariants."""

    # Verify operands.
    irdl_op_verify_arg_list(op, op_def, VariadicType.OPERAND)

    # Verify results.
    irdl_op_verify_arg_list(op, op_def, VariadicType.RESULT)

    # Verify regions.
    irdl_op_verify_arg_list(op, op_def, VariadicType.REGION)

    for idx, (region_name, region_def) in enumerate(op_def.regions):
        if isinstance(
                region_def,
                SingleBlockRegionDef) and len(op.regions[idx].blocks) != 1:
            raise VerifyException(
                f"region {region_name} at position {idx} should have a single block, but got {len(op.regions[idx].blocks)} blocks"
            )
        if region_def.block_args is not None:
            if len(op.regions[idx].blocks) == 0:
                raise VerifyException(
                    f"region {region_name} at position {idx} should have at least one block"
                )
            expected_num_args = len(region_def.block_args)
            num_args = len(op.regions[idx].blocks[0].args)
            if num_args != expected_num_args:
                raise VerifyException(
                    f"region {region_name} at position {idx} should have {expected_num_args} argument, but got {num_args}"
                )
            for arg_idx, arg_type in enumerate(op.regions[idx].blocks[0].args):
                typ = op.regions[idx].blocks[0].args[arg_idx]
                if arg_type != typ:
                    raise VerifyException(
                        f"argument at position {arg_idx} in region {region_name} at position {idx} should be of type {arg_type}, but {typ}"
                    )

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


def irdl_op_builder(cls: type[_OpT], op_def: OpDef,
                    operands: list[SSAValue | Operation
                                   | list[SSAValue | Operation]],
                    res_types: list[Any | list[Any]],
                    attributes: dict[str, Any], successors: Sequence[Block],
                    regions: Sequence[Region]) -> _OpT:
    """Builder for an irdl operation."""

    # We need irdl to define DenseIntOrFPElementsAttr, but here we need
    # DenseIntOrFPElementsAttr.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, i32)

    error_prefix = f"Error in {op_def.name} builder: "

    # Build operands by forwarding the values to SSAValue.get
    if len(op_def.operands) != len(operands):
        raise ValueError(
            error_prefix +
            f"expected {len(op_def.operands)} operands, got {len(operands)}")

    built_operands = list[SSAValue]()
    operand_variadic_sizes = list[int]()
    for ((operand_name, operand_def), operand) in zip(op_def.operands,
                                                      operands):
        if isinstance(operand_def, VarOperandDef):
            if not isinstance(operand, list):
                raise ValueError(
                    error_prefix +
                    f"'{operand_name}' operand: expected list argument for "
                    f"variadic operand, but got '{operand}' type.")
            built_operands.extend([SSAValue.get(arg) for arg in operand])
            operand_variadic_sizes.append(len(operand))
        else:
            if isinstance(operand, list):
                raise ValueError(
                    error_prefix +
                    f"'{operand_name}' operand: unexpected list argument for "
                    f"non-variadic operand.")
            built_operands.append(SSAValue.get(operand))

    # Build results by forwarding the values to the attribute builders
    if len(op_def.results) != len(res_types):
        raise ValueError(
            error_prefix +
            f"expected {len(op_def.results)} results, got {len(res_types)}")

    built_res_types = list[Attribute]()
    result_variadic_sizes = list[int]()
    for ((res_name, res_def), res_type) in zip(op_def.results, res_types):
        if isinstance(res_def, VarResultDef):
            if not isinstance(res_type, list):
                raise ValueError(
                    error_prefix +
                    f"'{res_name}' result: expected list argument for "
                    f"variadic result, but got '{res_type}' type.")
            built_res_types.extend([
                irdl_build_attribute(res_def.constr, res) for res in res_type
            ])
            result_variadic_sizes.append(len(res_type))
        else:
            built_res_types.append(
                irdl_build_attribute(res_def.constr, res_type))

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
        sizes = operand_variadic_sizes
        built_attributes[AttrSizedOperandSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    if AttrSizedResultSegments() in op_def.options:
        sizes = result_variadic_sizes
        built_attributes[AttrSizedResultSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    # Build regions using `Region.get`.
    regions = [Region.get(region) for region in regions]

    return cls.create(operands=built_operands,
                      result_types=built_res_types,
                      attributes=built_attributes,
                      successors=successors,
                      regions=regions)


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
    previous_variadics = 0
    for operand_idx, (operand_name, operand_def) in enumerate(op_def.operands):
        new_attrs[operand_name] = property(
            lambda self, idx=operand_idx, previous_vars=previous_variadics:
            get_operand_or_result(self, op_def, idx, previous_vars, True))
        if isinstance(operand_def, VarOperandDef):
            previous_variadics += 1
    if previous_variadics > 1 and AttrSizedOperandSegments(
    ) not in op_def.options:
        raise Exception(
            "Operation defines more than two variadic operands, "
            "but do not define the AttrSizedOperandSegments option")

    # Add result access fields
    previous_variadics = 0
    for result_idx, (result_name, result_def) in enumerate(op_def.results):
        new_attrs[result_name] = property(
            lambda self, idx=result_idx, previous_vars=previous_variadics:
            get_operand_or_result(self, op_def, idx, previous_vars, False))
        if isinstance(result_def, VarResultDef):
            previous_variadics += 1
    if previous_variadics > 1 and AttrSizedResultSegments(
    ) not in op_def.options:
        raise Exception("Operation defines more than two variadic results, "
                        "but do not define the AttrSizedResultSegments option")

    for region_idx, (region_name, _) in enumerate(op_def.regions):
        new_attrs[region_name] = property(
            lambda self, idx=region_idx: self.regions[idx])

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


def builder(f: BuilderTy[_AttrT]) -> BuilderTy[_AttrT]:
    """
    Annotate a function and mark it as an IRDL builder.
    This should only be used as decorator in classes decorated by irdl_attr_builder.
    """
    f.__irdl_is_builder = True
    return f


def irdl_get_builders(cls: type[_AttrT]) -> list[BuilderTy[_AttrT]]:
    """Get functions decorated with 'builder' in a class."""
    builders = list[BuilderTy[_AttrT]]()
    for field_name in cls.__dict__:
        field_ = cls.__dict__[field_name]
        # Builders are staticmethods, so we need to get back the original function with __func__
        if hasattr(field_, "__func__") and hasattr(field_.__func__,
                                                   "__irdl_is_builder"):
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
            f'"build" method for {cls.__name__} is reserved for IRDL, and should not be defined.'
        )
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
            if isinstance(expected_type, types.GenericAlias):
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
    for arg, param in zip(args, params[:num_non_defaults]):
        if not util.is_satisfying_hint(arg, param):
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
        if res is not None:
            return res
    raise TypeError(
        f"No available {cls.__name__} builders for arguments {args}")


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
            f'"build" method for {cls.__name__} is reserved for IRDL, and should not be defined.'
        )
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
        f"Class {cls.__name__} should either be a subclass of 'Data' or 'ParametrizedAttribute'"
    )
