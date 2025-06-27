import keyword

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.dialects.irdl import AnyOp
from xdsl.dialects.irdl.irdl import (
    AttributeOp,
    DialectOp,
    OperandsOp,
    OperationOp,
    ParametersOp,
    ResultsOp,
    VariadicityArrayAttr,
    VariadicityAttr,
)
from xdsl.ir import Attribute, Block, Dialect, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import (
    AttrConstraint,
    GenericRangeConstraint,
    IRDLOperation,
    OptionalDef,
    VariadicDef,
)


def depython_name(name: str):
    if name[-1] == "_" and keyword.iskeyword(name[:-1]):
        return name[:-1]
    return name


def constraint_to_irdl(builder: Builder, constraint: AttrConstraint) -> SSAValue:
    """
    Convert an attribute constraint to IRDL.
    This will create new operations at the provided builder location.
    """
    any_op = builder.insert(AnyOp())
    return any_op.output


def range_to_irdl(
    builder: Builder, constraint: GenericRangeConstraint[Attribute]
) -> SSAValue:
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
    builder = Builder(InsertPoint.at_end(block))

    # Operands
    operand_values: list[SSAValue] = []
    operand_variadicities: list[VariadicityAttr] = []
    operand_names: list[StringAttr] = []
    for operand in op_def.operands:
        operand_values.append(range_to_irdl(builder, operand[1].constr))
        if isinstance(operand[1], OptionalDef):
            operand_variadicities.append(VariadicityAttr.OPTIONAL)
        elif isinstance(operand[1], VariadicDef):
            operand_variadicities.append(VariadicityAttr.VARIADIC)
        else:
            operand_variadicities.append(VariadicityAttr.SINGLE)
        operand_names.append(StringAttr(depython_name(operand[0])))
    if operand_values:
        builder.insert(
            OperandsOp(
                operand_values,
                VariadicityArrayAttr(ArrayAttr(operand_variadicities)),
                ArrayAttr(operand_names),
            )
        )

    # Results
    result_values: list[SSAValue] = []
    result_variadicities: list[VariadicityAttr] = []
    result_names: list[StringAttr] = []
    for result in op_def.results:
        result_values.append(range_to_irdl(builder, result[1].constr))
        if isinstance(result[1], OptionalDef):
            result_variadicities.append(VariadicityAttr.OPTIONAL)
        elif isinstance(result[1], VariadicDef):
            result_variadicities.append(VariadicityAttr.VARIADIC)
        else:
            result_variadicities.append(VariadicityAttr.SINGLE)
        result_names.append(StringAttr(depython_name(result[0])))
    if result_values:
        builder.insert(
            ResultsOp(
                result_values,
                VariadicityArrayAttr(ArrayAttr(result_variadicities)),
                ArrayAttr(result_names),
            )
        )

    return OperationOp(Dialect.split_name(op_def.name)[1], Region([block]))


def attr_def_to_irdl(
    attr: type[ParametrizedAttribute],
) -> AttributeOp:
    """Convert an attribute definition to an IRDL attribute definition."""
    attr_def = attr.get_irdl_definition()

    block = Block()
    builder = Builder(InsertPoint.at_end(block))

    # Parameters
    param_values: list[SSAValue] = []
    names: list[StringAttr] = []
    for param in attr_def.parameters:
        param_values.append(constraint_to_irdl(builder, param[1]))
        names.append(StringAttr(depython_name(param[0])))
    builder.insert(ParametersOp(param_values, ArrayAttr(names)))

    return AttributeOp(Dialect.split_name(attr_def.name)[1], Region([block]))


def dialect_to_irdl(dialect: Dialect, name: str) -> DialectOp:
    """Convert a dialect definition to an IRDL dialect definition."""
    block = Block()
    builder = Builder(InsertPoint.at_end(block))

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
