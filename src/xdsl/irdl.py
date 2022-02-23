from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, TypeVar
from inspect import isclass
import typing

from xdsl.ir import Operation, Attribute, ParametrizedAttribute, SSAValue, Data, Region, Block
from xdsl import util


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
        if not isinstance(attr, Attribute):
            raise Exception(f"Expected attribute, but got {attr}")


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
class RegionDef(Region):
    """
    An IRDL region definition.
    If the block_args is specified, then the region expect to have the entry block with these arguments.
    """
    block_args: Optional[List[Attribute]] = None
    blocks: List[Block] = field(default_factory=list)


@dataclass
class SingleBlockRegionDef(RegionDef):
    """An IRDL region definition that expects exactly one block."""
    pass


@dataclass(init=False)
class AttributeDef:
    """An IRDL attribute definition."""

    constr: AttrConstraint
    """The attribute constraint."""

    data: typing.Any

    def __init__(self, typ: Union[Attribute, typing.Type[Attribute],
                                  AttrConstraint]):
        self.constr = attr_constr_coercion(typ)


@dataclass(init=False)
class OptAttributeDef(AttributeDef):
    """An IRDL attribute definition for an optional attribute."""

    def __init__(self, typ: Union[Attribute, typing.Type[Attribute],
                                  AttrConstraint]):
        super().__init__(typ)


def get_variadic_sizes(op: Operation, is_operand: bool) -> List[int]:
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
            raise Exception(
                f"Expected {size_attribute_name} attribute in {op.name} operation."
            )
        attribute = op.attributes[size_attribute_name]
        if not isinstance(attribute, DenseIntOrFPElementsAttr):
            raise Exception(
                f"{size_attribute_name} attribute is expected to be a DenseIntOrFPElementsAttr."
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
            if isinstance(attr_def, OptAttributeDef):
                continue
            raise Exception(f"attribute {attr_name} expected")
        attr_def.constr.verify(op.attributes[attr_name])


def irdl_build_attribute(irdl_def: AttrConstraint, result) -> Attribute:
    if isinstance(irdl_def, BaseAttr):
        if isinstance(result, Tuple):
            return irdl_def.attr.build(*result)
        return irdl_def.attr.build(result)
    if isinstance(result, Attribute):
        return result
    raise Exception(f"builder expected an attribute, got {result}")


OpT = TypeVar('OpT', bound='Operation')


def irdl_op_builder(cls: typing.Type[OpT], operands: List,
                    operand_defs: List[Tuple[str, OperandDef]],
                    res_types: List, res_defs: List[Tuple[str, ResultDef]],
                    attributes: typing.Dict[str, typing.Any],
                    attr_defs: typing.Dict[str, AttributeDef], successors,
                    regions, options) -> OpT:
    """Builder for an irdl operation."""

    # We need irdl to define DenseIntOrFPElementsAttr, but here we need
    # DenseIntOrFPElementsAttr.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, VectorType, IntegerType, i32

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
            len(operand)
            for operand, (_, operand_def) in zip(operands, operand_defs)
            if isinstance(operand_def, VarOperandDef)
        ]
        built_attributes[AttrSizedOperandSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    if AttrSizedResultSegments() in options:
        sizes = [
            len(result)
            for result, (_, result_def) in zip(res_types, res_defs)
            if isinstance(result_def, VarResultDef)
        ]
        built_attributes[AttrSizedResultSegments.attribute_name] =\
            DenseIntOrFPElementsAttr.vector_from_list(sizes, i32)

    # Build regions using `Region.get`.
    regions = [Region.get(region) for region in regions]

    return cls.create(operands=built_operands,
                      result_types=built_res_types,
                      attributes=built_attributes,
                      successors=successors,
                      regions=regions)


OperationType = TypeVar("OperationType", bound=Operation)


def irdl_op_definition(
        cls: typing.Type[OperationType]) -> typing.Type[OperationType]:
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


C = TypeVar('C', bound='Callable')


def builder(f: C) -> C:
    """
    Annotate a function and mark it as an IRDL builder.
    This should only be used as decorator in classes decorated by irdl_attr_builder.
    """
    f.__irdl_is_builder = True
    return f


def irdl_get_builders(cls) -> List[typing.Callable]:
    builders = []
    for field_name in cls.__dict__:
        field_ = cls.__dict__[field_name]
        # Builders are staticmethods, so we need to get back the original function with __func__
        if hasattr(field_, "__func__") and hasattr(field_.__func__,
                                                   "__irdl_is_builder"):
            builders.append(field_.__func__)
    return builders


def irdl_attr_try_builder(builder, *args):
    params_dict = typing.get_type_hints(builder)
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


T = TypeVar('T')


def irdl_data_definition(cls: typing.Type[T]) -> typing.Type[T]:
    builders = irdl_get_builders(cls)
    if "build" in cls.__dict__:
        raise Exception(
            f'"build" method for {cls.__name__} is reserved for IRDL, and should not be defined.'
        )
    new_attrs = dict()
    new_attrs["build"] = lambda *args: irdl_attr_builder(cls, builders, *args)
    return dataclass(frozen=True)(type(cls.__name__, (cls, ), {
        **cls.__dict__,
        **new_attrs
    }))


AttributeType = TypeVar("AttributeType", bound=ParametrizedAttribute)


def irdl_param_attr_definition(
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

    builders = irdl_get_builders(cls)
    if "build" in cls.__dict__:
        raise Exception(
            f'"build" method for {cls.__name__} is reserved for IRDL, and should not be defined.'
        )
    new_attrs["build"] = lambda *args: irdl_attr_builder(cls, builders, *args)

    return dataclass(frozen=True)(type(cls.__name__, (cls, ), {
        **cls.__dict__,
        **new_attrs
    }))


def irdl_attr_definition(cls: typing.Type[T]) -> typing.Type[T]:
    if issubclass(cls, ParametrizedAttribute):
        return irdl_param_attr_definition(cls)
    if issubclass(cls, Data):
        return irdl_data_definition(cls)
    raise Exception(
        f"Class {cls.__name__} should either be a subclass of 'Data' or 'ParametrizedAttribute'"
    )
