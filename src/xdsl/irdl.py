from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod

from xdsl.ir import Operation, Attribute, ParametrizedAttribute, SSAValue
from typing import List, Tuple, Optional, Union, TypeVar
from inspect import isclass
import typing


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
            raise Exception(f"Expected attribute {self.attr} but got {attr}")


@dataclass
class BaseAttr(AttrConstraint):
    """Constrain an attribute to be of a given base type."""

    attr: typing.Type[Attribute]
    """The expected attribute base type."""
    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, self.attr):
            raise Exception(
                f"{attr} should be of base attribute {self.attr.name}")


def attr_constr_coercion(
    attr: Union[Attribute, typing.Type[Attribute], AttrConstraint]
) -> AttrConstraint:
    """
    Attributes are coerced into EqAttrConstraints,
    and Attribute types are coerced into BaseAttr.
    """
    if isinstance(attr, Attribute):
        return EqAttrConstraint(attr)
    if isclass(attr) and issubclass(attr, Attribute):
        return BaseAttr(attr)
    return attr


@dataclass
class AnyAttr(AttrConstraint):
    """Constraint that is verified by all attributes."""
    def verify(self, attr: Attribute) -> None:
        pass


@dataclass(init=False)
class AnyOf(AttrConstraint):
    """Ensure that an attribute satisfies one of the given constraints."""

    attr_constrs: List[AttrConstraint]
    """The list of constraints that are checked."""
    def __init__(self,
                 attr_constrs: List[Union[Attribute, typing.Type[Attribute],
                                          AttrConstraint]]):
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
            except Exception:
                pass
        raise Exception(f"Unexpected attribute {attr}")


@dataclass(init=False)
class ParamAttrConstraint(AttrConstraint):
    """
    Constrain an attribute to be of a given type,
    and also constrain its parameters with additional constraints.
    """

    base_attr: typing.Type[Attribute]
    """The base attribute type."""

    param_constrs: List[AttrConstraint]
    """The attribute parameter constraints"""
    def __init__(self, base_attr: typing.Type[Attribute],
                 param_constrs: List[Union[Attribute, typing.Type[Attribute],
                                           AttrConstraint]]):
        self.base_attr = base_attr
        self.param_constrs = [
            attr_constr_coercion(constr) for constr in param_constrs
        ]

    def verify(self, attr: Attribute) -> None:
        assert isinstance(attr, ParametrizedAttribute)
        if not isinstance(attr, self.base_attr):
            raise Exception(
                f"Base attribute {self.base_attr.name} expected, but got {attr.name}"
            )
        if len(self.param_constrs) != len(attr.parameters):
            raise Exception(
                f"{len(self.param_constrs)} parameters expected, but got {len(attr.parameters)}"
            )
        for idx, param_constr in enumerate(self.param_constrs):
            param_constr.verify(attr.parameters[idx])


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
    def __init__(self, typ: Union[Attribute, typing.Type[Attribute],
                                  AttrConstraint]):
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
    def __init__(self, typ: Union[Attribute, typing.Type[Attribute],
                                  AttrConstraint]):
        self.constr = attr_constr_coercion(typ)


@dataclass(init=False)
class VarResultDef(ResultDef, VariadicDef):
    """An IRDL variadic result definition."""


@dataclass(init=False)
class OptResultDef(VarResultDef, OptionalDef):
    """An IRDL optional result definition."""


@dataclass
class RegionDef:
    """
    An IRDL region definition.
    If the block_args is specified, then the region expect to have the entry block with these arguments.
    """
    block_args: Optional[List[Attribute]] = None


@dataclass
class SingleBlockRegionDef(RegionDef):
    """An IRDL region definition that expects exactly one block."""
    pass


@dataclass(init=False)
class AttributeDef:
    """An IRDL attribute definition."""

    constr: AttrConstraint
    """The attribute constraint."""
    def __init__(self, typ: Union[Attribute, typing.Type[Attribute],
                                  AttrConstraint]):
        self.constr = attr_constr_coercion(typ)


def get_variadic_sizes(op: Operation, is_operand: bool) -> List[int]:
    """Get variadic sizes of operands or results."""

    # We need irdl to define VectorAttr, but here we need VectorAttr.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import VectorAttr

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
            raise Exception(
                f"Expected {size_attribute_name} attribute in {op.name} operation."
            )
        attribute = op.attributes[size_attribute_name]
        if not isinstance(attribute, VectorAttr):
            raise Exception(
                f"{size_attribute_name} attribute is expected to be a VectorAttr."
            )
        variadic_sizes = [
            size_attr.value.data for size_attr in attribute.data.data
        ]
        if len(variadic_sizes) != len(variadic_defs):
            raise Exception(
                f"expected {len(variadic_defs)} values in {size_attribute_name}, but got {len(variadic_sizes)}"
            )
        if len(operand_or_result_defs) - len(variadic_defs) + sum(
                variadic_sizes) != len(op_defs):
            raise Exception(
                f"{size_attribute_name} values does not correspond to variadic arguments sizes."
            )
        return variadic_sizes

    # If there are no variadics arguments, we just check that we have the right number of arguments
    if len(variadic_defs) == 0:
        if len(op_defs) != len(operand_or_result_defs):
            raise Exception(
                f"Expected {len(operand_or_result_defs)} {'operands' if is_operand else 'results'}, but got {len(op_defs)}"
            )
        return []

    # If there is a single variadic argument, we can get its size from the number of arguments.
    if len(variadic_defs) == 1:
        if len(op_defs) - len(operand_or_result_defs) + 1 < 0:
            raise Exception(
                f"Expected at least {len(operand_or_result_defs) - 1} {def_type_name}s, got {len(operand_or_result_defs)}"
            )
        return [len(op_defs) - len(operand_or_result_defs) + 1]

    # Unreachable, all cases should have been handled.
    # Additional cases should raise an exception upon definition of the irdl operation.
    assert False


def get_operand_or_result(
        op: Operation, arg_def_idx: int, previous_var_args: int,
        is_operand: bool
) -> Union[SSAValue, Optional[SSAValue], List[SSAValue]]:
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


def irdl_op_verify(op: Operation, operands: List[Tuple[str, OperandDef]],
                   results: List[Tuple[str, ResultDef]],
                   regions: List[Tuple[str, RegionDef]],
                   attributes: List[Tuple[str, AttributeDef]]) -> None:
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
            current_operand += 1
        else:
            operand_def.constr.verify(op.operands[current_operand].typ)
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
            current_result += 1
        else:
            result_def.constr.verify(op.results[current_result].typ)
            current_result += 1

    if len(regions) != len(op.regions):
        raise Exception(
            f"op has {len(op.regions)} regions, but {len(regions)} were expected"
        )

    for idx, (region_name, region_def) in enumerate(regions):
        if isinstance(
                region_def,
                SingleBlockRegionDef) and len(op.regions[idx].blocks) != 1:
            raise Exception(
                f"region {region_name} at position {idx} should have a single block, but got {len(op.regions[idx].blocks)} blocks"
            )
        if region_def.block_args is not None:
            if len(op.regions[idx].blocks) == 0:
                raise Exception(
                    f"region {region_name} at position {idx} should have at least one block"
                )
            expected_num_args = len(region_def.block_args)
            num_args = len(op.regions[idx].blocks[0].args)
            if num_args != expected_num_args:
                raise Exception(
                    f"region {region_name} at position {idx} should have {expected_num_args} argument, but got {num_args}"
                )
            for arg_idx, arg_type in enumerate(op.regions[idx].blocks[0].args):
                typ = op.regions[idx].blocks[0].args[arg_idx]
                if arg_type != typ:
                    raise Exception(
                        f"argument at position {arg_idx} in region {region_name} at position {idx} should be of type {arg_type}, but {typ}"
                    )

    for attr_name, attr_def in attributes:
        if attr_name not in op.attributes:
            raise Exception(f"attribute {attr_name} expected")
        attr_def.constr.verify(op.attributes[attr_name])


OperationType = TypeVar("OperationType", bound=Operation)


def irdl_op_definition(
        cls: typing.Type[OperationType]) -> typing.Type[OperationType]:
    """Decorator used on classes to define a new operation definition."""
    operands = [(field_name, field)
                for field_name, field in cls.__dict__.items()
                if isinstance(field, OperandDef)]
    results = [(field_name, field)
               for field_name, field in cls.__dict__.items()
               if isinstance(field, ResultDef)]
    regions = [(field_name, field)
               for field_name, field in cls.__dict__.items()
               if isinstance(field, RegionDef)]
    attributes = [(field_name, field)
                  for field_name, field in cls.__dict__.items()
                  if isinstance(field, AttributeDef)]
    options = cls.__dict__.get("irdl_options", [])
    new_attrs = dict()

    # Add operand access fields
    previous_variadics = 0
    for operand_idx, (operand_name, operand_def) in enumerate(operands):
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
    for result_idx, (result_name, result_def) in enumerate(results):
        new_attrs[result_name] = property(
            lambda self, idx=result_idx, previous_vars=previous_variadics:
            get_operand_or_result(self, idx, previous_vars, False))
        if isinstance(result_def, VarResultDef):
            previous_variadics += 1
    if previous_variadics > 1 and AttrSizedResultSegments() not in options:
        raise Exception("Operation defines more than two variadic results, "
                        "but do not define the AttrSizedResultSegments option")

    for region_idx, (region_name, _) in enumerate(regions):
        new_attrs[region_name] = property(
            (lambda idx: lambda self: self.regions[idx])(region_idx))

    for attribute_name, _ in attributes:
        new_attrs[attribute_name] = property(
            (lambda name: lambda self: self.attributes[name])(attribute_name))

    new_attrs["irdl_operand_defs"] = operands
    new_attrs["irdl_result_defs"] = results
    new_attrs["irdl_region_defs"] = regions
    new_attrs["irdl_attribute_defs"] = attributes
    new_attrs["irdl_options"] = options

    new_attrs["verify_"] = lambda op: irdl_op_verify(op, operands, results,
                                                     regions, attributes)
    if "verify_" in cls.__dict__:
        custom_verifier = cls.__dict__["verify_"]

        def new_verifier(verifier, op):
            verifier(op)
            custom_verifier(op)

        new_attrs["verify_"] = (
            lambda verifier: lambda op: new_verifier(verifier, op))(
                new_attrs["verify_"])

    return type(cls.__name__, (Operation, ), {**cls.__dict__, **new_attrs})


@dataclass
class ParameterDef:
    """An IRDL definition of an attribute parameter."""
    constr: AttrConstraint

    def __init__(self, typ: Union[Attribute, typing.Type[Attribute],
                                  AttrConstraint]):
        self.constr = attr_constr_coercion(typ)


def irdl_attr_verify(attr: ParametrizedAttribute,
                     parameters: List[ParameterDef]):
    """Given an IRDL definition, verify that an attribute satisfies its invariants."""

    if len(attr.parameters) != len(parameters):
        raise Exception(
            f"{len(parameters)} parameters expected, got {len(attr.parameters)}"
        )
    for idx, param_def in enumerate(parameters):
        param_def.constr.verify(attr.parameters[idx])


AttributeType = TypeVar("AttributeType", bound=ParametrizedAttribute)


def irdl_attr_definition(
        cls: typing.Type[AttributeType]) -> typing.Type[AttributeType]:
    """Decorator used on classes to define a new attribute definition."""

    parameters = []
    new_attrs = dict()
    for field_name in cls.__dict__:
        field_ = cls.__dict__[field_name]
        if isinstance(field_, ParameterDef):
            new_attrs[field_name] = property(
                (lambda idx: lambda self: self.parameters[idx])(
                    len(parameters)))
            parameters.append(field_)

    new_attrs["verify"] = lambda typ: irdl_attr_verify(typ, parameters)

    if "verify" in cls.__dict__:
        custom_verifier = cls.__dict__["verify"]

        def new_verifier(verifier, op):
            verifier(op)
            custom_verifier(op)

        new_attrs["verify"] = (
            lambda verifier: lambda op: new_verifier(verifier, op))(
                new_attrs["verify"])

    return type(cls.__name__, (ParametrizedAttribute, ), {
        **cls.__dict__,
        **new_attrs
    })
