from xdsl.builder import Builder
from xdsl.dialects.irdl import AnyOp
from xdsl.dialects.irdl.irdl import (
    AttributeOp,
    DialectOp,
    OperandsOp,
    OperationOp,
    ParametersOp,
    ResultsOp,
    VariadicityAttr,
)
from xdsl.ir import Attribute, Block, Dialect, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import AttrConstraint, IRDLOperation, RangeConstraint


def constraint_to_irdl(builder: Builder, constraint: AttrConstraint) -> SSAValue:
    """
    Convert an attribute constraint to IRDL.
    This will create new operations at the provided builder location.
    """
    any_op = builder.insert(AnyOp())
    return any_op.output


def range_to_irdl(builder: Builder, constraint: RangeConstraint[Attribute]) -> SSAValue:
    """
    Convert a range constraint to IRDL.
    This will create new operations at the provided builder location.
    """
    any_op = builder.insert(AnyOp())
    return any_op.output


def op_def_to_irdl(op: type[IRDLOperation]) -> OperationOp:
    """Convert an operation definition to an IRDL operation definition."""
    op_def = op.get_irdl_definition()

    block = Block()
    builder = Builder.at_end(block)

    # Operands
    operand_values: dict[str, tuple[VariadicityAttr, SSAValue]] = {}
    for name, operand in op_def.operands:
        operand_values[name] = (
            VariadicityAttr.SINGLE,
            range_to_irdl(builder, operand.constr),
        )
    if operand_values:
        builder.insert(OperandsOp.get(operand_values))

    # Results
    result_values: dict[str, tuple[VariadicityAttr, SSAValue]] = {}
    for name, result in op_def.results:
        result_values[name] = (
            VariadicityAttr.SINGLE,
            range_to_irdl(builder, result.constr),
        )
    if result_values:
        builder.insert(ResultsOp.get(result_values))

    return OperationOp(Dialect.split_name(op_def.name)[1], Region([block]))


def attr_def_to_irdl(
    attr: type[ParametrizedAttribute],
) -> AttributeOp:
    """Convert an attribute definition to an IRDL attribute definition."""
    attr_def = attr.get_irdl_definition()

    block = Block()
    builder = Builder.at_end(block)

    # Parameters
    param_values: list[SSAValue] = []
    for param in attr_def.parameters:
        param_values.append(constraint_to_irdl(builder, param[1]))
    builder.insert(ParametersOp(param_values))

    return AttributeOp(Dialect.split_name(attr_def.name)[1], Region([block]))


def dialect_to_irdl(dialect: Dialect, name: str) -> DialectOp:
    """Convert a dialect definition to an IRDL dialect definition."""
    block = Block()
    builder = Builder.at_end(block)

    for attribute in dialect.attributes:
        if not issubclass(attribute, ParametrizedAttribute):
            raise ValueError(
                "Can only convert ParametrizedAttribute attributes to IRDL"
            )
        builder.insert(attr_def_to_irdl(attribute))

    for operation in dialect.operations:
        if not issubclass(operation, IRDLOperation):
            raise ValueError("Can only convert IRDLOperations operations to IRDL")
        builder.insert(op_def_to_irdl(operation))

    return DialectOp(name, Region([block]))
