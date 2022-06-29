from __future__ import annotations

import inspect
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from inspect import isclass
from typing import (Annotated, Any, Callable, Dict, Generic, List, Optional,
                    Sequence, Tuple, Type, TypeAlias, TypeVar, Union, cast,
                    get_args, get_origin, get_type_hints)

from xdsl import util
from xdsl.diagnostic import Diagnostic, DiagnosticException
from xdsl.ir import (Attribute, Block, Data, Operation, ParametrizedAttribute,
                     Region, SSAValue)


def error(op: Operation, msg: str):
    diag = Diagnostic()
    diag.add_message(op, msg)
    diag.raise_exception(f"{op.name} operation does not verify", op)


class VerifyException(DiagnosticException):
    ...


class IRDLAnnotations(Enum):
    ParamDefAnnot = 1


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
class SingleBlockRegionDef(RegionDef):
    """An IRDL region definition that expects exactly one block."""
    pass


@dataclass(init=False)
class AttributeDef:
    """An IRDL attribute definition."""

    constr: AttrConstraint
    """The attribute constraint."""

    data: Any

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        self.constr = attr_constr_coercion(typ)


@dataclass(init=False)
class OptAttributeDef(AttributeDef):
    """An IRDL attribute definition for an optional attribute."""

    def __init__(self, typ: Attribute | type[Attribute] | AttrConstraint):
        super().__init__(typ)


def get_variadic_sizes(op: Operation, is_operand: bool) -> list[int]:
    """Get variadic sizes of operands or results."""

    # We need irdl to define DenseIntOrFPElementsAttr, but here we need
    # DenseIntOrFPElementsAttr.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    operand_or_result_defs = op.irdl_operand_defs if is_operand else op.irdl_result_defs
    variadic_defs = [(arg_name, arg_def)
                     for arg_name, arg_def in operand_or_result_defs
                     if isinstance(arg_def, VariadicDef)]
    options = op.irdl_options

    op_defs = op.operands if is_operand else op.results
    def_type_name = "operand" if is_operand else "result"

    # If the size is in the attributes, fetch it
    attribute_option = AttrSizedOperandSegments(
    ) if is_operand else AttrSizedResultSegments()
    if attribute_option in options:
        size_attribute_name = AttrSizedOperandSegments.attribute_name if is_operand else AttrSizedResultSegments.attribute_name
        if size_attribute_name not in op.attributes:
            raise VerifyException(
                f"Expected {size_attribute_name} attribute in {op.name} operation."
            )
        attribute = op.attributes[size_attribute_name]
        if not isinstance(attribute, DenseIntOrFPElementsAttr):
            raise VerifyException(
                f"{size_attribute_name} attribute is expected to be a DenseIntOrFPElementsAttr."
            )
        variadic_sizes = [
            size_attr.value.data for size_attr in attribute.data.data
        ]
        if len(variadic_sizes) != len(operand_or_result_defs):
            raise VerifyException(
                f"expected {len(operand_or_result_defs)} values in {size_attribute_name}, but got {len(variadic_sizes)}"
            )
        return variadic_sizes

    # If there are no variadics arguments, we just check that we have the right number of arguments
    if len(variadic_defs) == 0:
        if len(op_defs) != len(operand_or_result_defs):
            raise VerifyException(
                f"Expected {len(operand_or_result_defs)} {'operands' if is_operand else 'results'}, but got {len(op_defs)}"
            )
        return []

    # If there is a single variadic argument, we can get its size from the number of arguments.
    if len(variadic_defs) == 1:
        if len(op_defs) - len(operand_or_result_defs) + 1 < 0:
            raise VerifyException(
                f"Expected at least {len(operand_or_result_defs) - 1} {def_type_name}s, got {len(operand_or_result_defs)}"
            )
        return [len(op_defs) - len(operand_or_result_defs) + 1]

    # Unreachable, all cases should have been handled.
    # Additional cases should raise an exception upon definition of the irdl operation.
    assert False


def get_operand_or_result(
        op: Operation, arg_def_idx: int, previous_var_args: int,
        is_operand: bool) -> SSAValue | None | list[SSAValue]:
    """
    Get an operand or a result.
    In the case of a variadic operand or result definition, return a list of operand or results.
    :param op: The operation we want to get the operand or result of.
    :param arg_def_idx: The operand or result index in the irdl definition.
    :param previous_var_args: The number of previous variadic operands or results definition before this definition.
    :param is_operand: Do we get the operand or the result.
    :return:
    """
    argument_defs = op.irdl_operand_defs if is_operand else op.irdl_result_defs
    op_arguments = op.operands if is_operand else op.results

    variadic_sizes = get_variadic_sizes(op, is_operand)

    begin_arg = arg_def_idx - previous_var_args + sum(
        variadic_sizes[:previous_var_args])
    if isinstance(argument_defs[arg_def_idx][1], OptionalDef):
        arg_size = variadic_sizes[previous_var_args]
        if arg_size == 0:
            return None
        else:
            return op_arguments[begin_arg]
    if isinstance(argument_defs[arg_def_idx][1], VariadicDef):
        arg_size = variadic_sizes[previous_var_args]
        return op_arguments[begin_arg:begin_arg + arg_size]
    else:
        return op_arguments[begin_arg]


def irdl_op_verify(op: Operation, operands: list[tuple[str, OperandDef]],
                   results: list[tuple[str, ResultDef]],
                   regions: list[tuple[str, RegionDef]],
                   attributes: list[tuple[str, AttributeDef]]) -> None:
    """Given an IRDL definition, verify that an operation satisfies its invariants."""

    # Verify operands.
    # get_variadic_sizes already verify that the variadic operand sizes match the number of operands.
    operand_sizes = get_variadic_sizes(op, is_operand=True)
    current_operand = 0
    current_var_operand = 0
    for operand_name, operand_def in operands:
        if isinstance(operand_def, VarOperandDef):
            for i in range(operand_sizes[current_var_operand]):
                operand_def.constr.verify(op.operands[current_operand].typ)
                current_operand += 1
        else:
            try:
                operand_def.constr.verify(op.operands[current_operand].typ)
            except Exception as e:
                error(
                    op,
                    f"Operand {operand_name} at operand position {current_operand} (counted from zero) does not verify!\n{e}"
                )

            current_operand += 1

    # Verify results
    # get_variadic_sizes already verify that the variadic result sizes match the number of results.
    result_sizes = get_variadic_sizes(op, is_operand=False)
    current_result = 0
    current_var_result = 0
    for result_name, result_def in results:
        if isinstance(result_def, VarResultDef):
            for i in range(result_sizes[current_var_result]):
                result_def.constr.verify(op.results[current_result].typ)
                current_result += 1
        else:
            result_def.constr.verify(op.results[current_result].typ)
            current_result += 1

    if len(regions) != len(op.regions):
        raise VerifyException(
            f"op has {len(op.regions)} regions, but {len(regions)} were expected"
        )

    for idx, (region_name, region_def) in enumerate(regions):
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

    for attr_name, attr_def in attributes:
        if attr_name not in op.attributes:
            if isinstance(attr_def, OptAttributeDef):
                continue
            raise VerifyException(f"attribute {attr_name} expected")
        attr_def.constr.verify(op.attributes[attr_name])


def irdl_build_attribute(irdl_def: AttrConstraint, result) -> Attribute:
    if isinstance(irdl_def, BaseAttr):
        if isinstance(result, tuple):
            return irdl_def.attr.build(*result)
        return irdl_def.attr.build(result)
    if isinstance(result, Attribute):
        return result
    raise Exception(f"builder expected an attribute, got {result}")


OpT = TypeVar('OpT', bound=Operation)


def irdl_op_builder(cls: type[OpT], operands: list[Any],
                    operand_defs: list[tuple[str, OperandDef]],
                    res_types: list[Any], res_defs: list[tuple[str,
                                                               ResultDef]],
                    attributes: dict[str, Any], attr_defs: dict[str,
                                                                AttributeDef],
                    successors, regions, options) -> OpT:
    """Builder for an irdl operation."""

    # We need irdl to define DenseIntOrFPElementsAttr, but here we need
    # DenseIntOrFPElementsAttr.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import (DenseIntOrFPElementsAttr, IntegerAttr,
                                       IntegerType, VectorType, i32)

    # Build operands by forwarding the values to SSAValue.get
    if len(operand_defs) != len(operands):
        raise ValueError(
            f"Expected {len(operand_defs)} operands, got {len(operands)}")

    built_operands = []
    for ((_, operand_def), operand) in zip(operand_defs, operands):
        if isinstance(operand_def, VarOperandDef):
            if not isinstance(operand, list):
                raise ValueError(
                    f"Expected list for variadic operand builder, got {operand}"
                )
            built_operands.extend([SSAValue.get(arg) for arg in operand])
        else:
            built_operands.append(SSAValue.get(operand))

    # Build results by forwarding the values to the attribute builders
    if len(res_defs) != len(res_types):
        raise ValueError(
            f"Expected {len(res_defs)} results, got {len(res_types)}")

    built_res_types = []
    for ((_, res_def), res_type) in zip(res_defs, res_types):
        if isinstance(res_def, VarResultDef):
            if not isinstance(res_type, list):
                raise ValueError(
                    f"Expected list for variadic result builder, got {res_type}"
                )
            built_res_types.extend([
                irdl_build_attribute(res_def.constr, res) for res in res_type
            ])
        else:
            built_res_types.append(
                irdl_build_attribute(res_def.constr, res_type))

    # Build attributes by forwarding the values to the attribute builders
    attr_defs = {name: def_ for (name, def_) in attr_defs}
    built_attributes = dict()
    for attr_name, attr in attributes.items():
        if attr_name not in attr_defs:
            if isinstance(attr, Attribute):
                built_attributes[attr_name] = attr
                continue
            raise ValueError(
                f"Unexpected attribute name {attr_name} for operation {cls.name}"
            )
        built_attributes[attr_name] = irdl_build_attribute(
            attr_defs[attr_name].constr, attr)

    # Take care of variadic operand and result segment sizes.
    if AttrSizedOperandSegments() in options:
        sizes = [
            (len(operand) if isinstance(operand_def, VarOperandDef) else 1)
            for operand, (_, operand_def) in zip(operands, operand_defs)
        ]
        built_attributes[AttrSizedOperandSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    if AttrSizedResultSegments() in options:
        sizes = [(len(result) if isinstance(result_def, VarResultDef) else 1)
                 for result, (_, result_def) in zip(res_types, res_defs)]
        built_attributes[AttrSizedResultSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    # Build regions using `Region.get`.
    regions = [Region.get(region) for region in regions]

    return cls.create(operands=built_operands,
                      result_types=built_res_types,
                      attributes=built_attributes,
                      successors=successors,
                      regions=regions)


def irdl_op_definition(cls: type[OpT]) -> type[OpT]:
    """Decorator used on classes to define a new operation definition."""

    assert issubclass(
        cls,
        Operation), f"class {cls.__name__} should be a subclass of Operation"

    # Get all fields of the class, including the parent classes
    clsdict = dict()
    for parent_cls in cls.mro()[::-1]:
        clsdict = {**clsdict, **parent_cls.__dict__}

    operand_defs = [(field_name, field)
                    for field_name, field in clsdict.items()
                    if isinstance(field, OperandDef)]
    result_defs = [(field_name, field)
                   for field_name, field in clsdict.items()
                   if isinstance(field, ResultDef)]
    region_defs = [(field_name, field)
                   for field_name, field in clsdict.items()
                   if isinstance(field, RegionDef)]
    attr_defs = [(field_name, field) for field_name, field in clsdict.items()
                 if isinstance(field, AttributeDef)]
    options = clsdict.get("irdl_options", [])
    new_attrs = dict()

    # Add operand access fields
    previous_variadics = 0
    for operand_idx, (operand_name, operand_def) in enumerate(operand_defs):
        new_attrs[operand_name] = property(
            lambda self, idx=operand_idx, previous_vars=previous_variadics:
            get_operand_or_result(self, idx, previous_vars, True))
        if isinstance(operand_def, VarOperandDef):
            previous_variadics += 1
    if previous_variadics > 1 and AttrSizedOperandSegments() not in options:
        raise Exception(
            "Operation defines more than two variadic operands, "
            "but do not define the AttrSizedOperandSegments option")

    # Add result access fields
    previous_variadics = 0
    for result_idx, (result_name, result_def) in enumerate(result_defs):
        new_attrs[result_name] = property(
            lambda self, idx=result_idx, previous_vars=previous_variadics:
            get_operand_or_result(self, idx, previous_vars, False))
        if isinstance(result_def, VarResultDef):
            previous_variadics += 1
    if previous_variadics > 1 and AttrSizedResultSegments() not in options:
        raise Exception("Operation defines more than two variadic results, "
                        "but do not define the AttrSizedResultSegments option")

    for region_idx, (region_name, _) in enumerate(region_defs):
        new_attrs[region_name] = property(
            (lambda idx: lambda self: self.regions[idx])(region_idx))

    for attribute_name, attr_def in attr_defs:
        if isinstance(attr_def, OptAttributeDef):
            new_attrs[attribute_name] = property(
                (lambda name: lambda self: self.attributes.get(name, None)
                 )(attribute_name))
        else:
            new_attrs[attribute_name] = property(
                (lambda name: lambda self: self.attributes[name]
                 )(attribute_name))

    new_attrs["irdl_operand_defs"] = operand_defs
    new_attrs["irdl_result_defs"] = result_defs
    new_attrs["irdl_region_defs"] = region_defs
    new_attrs["irdl_attribute_defs"] = attr_defs
    new_attrs["irdl_options"] = options

    new_attrs["verify_"] = lambda op: irdl_op_verify(
        op, operand_defs, result_defs, region_defs, attr_defs)
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
        return irdl_op_builder(cls, operands, operand_defs, result_types,
                               result_defs, attributes, attr_defs, successors,
                               regions, options)

    new_attrs["build"] = classmethod(builder)

    return type(cls.__name__, cls.__mro__, {**cls.__dict__, **new_attrs})


_A = TypeVar("_A", bound=Attribute)

ParameterDef: TypeAlias = Annotated[_A, IRDLAnnotations.ParamDefAnnot]


def irdl_attr_verify(attr: ParametrizedAttribute,
                     parameters: list[AttrConstraint]):
    """Given an IRDL definition, verify that an attribute satisfies its invariants."""

    if len(attr.parameters) != len(parameters):
        raise VerifyException(
            f"{len(parameters)} parameters expected, got {len(attr.parameters)}"
        )
    for idx, param_def in enumerate(parameters):
        param = attr.parameters[idx]
        param_def.verify(param)


C = TypeVar('C', bound=Callable[..., Any])


def builder(f: C) -> C:
    """
    Annotate a function and mark it as an IRDL builder.
    This should only be used as decorator in classes decorated by irdl_attr_builder.
    """
    f.__irdl_is_builder = True
    return f


def irdl_get_builders(cls) -> list[Callable[..., Any]]:
    builders = []
    for field_name in cls.__dict__:
        field_ = cls.__dict__[field_name]
        # Builders are staticmethods, so we need to get back the original function with __func__
        if hasattr(field_, "__func__") and hasattr(field_.__func__,
                                                   "__irdl_is_builder"):
            builders.append(field_.__func__)
    return builders


def irdl_attr_try_builder(builder, *args):
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


def irdl_attr_builder(cls, builders, *args):
    """Try to apply all builders to construct an attribute instance."""
    if len(args) == 1 and isinstance(args[0], cls):
        return args[0]
    for builder in builders:
        res = irdl_attr_try_builder(builder, *args)
        if res is not None:
            return res
    raise TypeError(
        f"No available {cls.__name__} builders for arguments {args}")


def irdl_data_verify(data: Data, typ: type) -> None:
    """Check that the Data has the expected type."""
    if isinstance(data.data, typ):
        return
    raise VerifyException(
        f"{data.name} data attribute expected type {typ}, but {type(data.data)} given."
    )


T = TypeVar('T')


def irdl_data_definition(cls: type[T]) -> type[T]:
    new_attrs = dict()

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


def irdl_param_attr_get_param_type_hints(
        cls: type[ParametrizedAttribute]) -> list[tuple[str, Any]]:
    """Get the type hints of an IRDL parameter definitions."""
    res = []
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


PA = TypeVar("PA", bound=ParametrizedAttribute)


def irdl_param_attr_definition(cls: type[PA]) -> type[PA]:
    """Decorator used on classes to define a new attribute definition."""

    # Get the fields from the class and its parents
    clsdict = dict()
    for parent_cls in cls.mro()[::-1]:
        clsdict = {**clsdict, **parent_cls.__dict__}

    param_hints = irdl_param_attr_get_param_type_hints(cls)

    # IRDL parameters definitions
    parameters = []
    # New fields and methods added to the attribute
    new_fields = dict()

    for param_name, param_type in param_hints:
        new_fields[param_name] = property(
            (lambda idx: lambda self: self.parameters[idx])(len(parameters)))
        parameters.append(
            irdl_to_attr_constraint(param_type, allow_type_var=True))

    new_fields["verify"] = lambda typ: irdl_attr_verify(typ, parameters)

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

    return dataclass(frozen=True, init=False)(type(cls.__name__, (cls, ), {
        **cls.__dict__,
        **new_fields
    }))


def irdl_attr_definition(cls: type[T]) -> type[T]:
    if issubclass(cls, ParametrizedAttribute):
        return irdl_param_attr_definition(cls)
    if issubclass(cls, Data):
        return irdl_data_definition(cls)
    raise Exception(
        f"Class {cls.__name__} should either be a subclass of 'Data' or 'ParametrizedAttribute'"
    )
