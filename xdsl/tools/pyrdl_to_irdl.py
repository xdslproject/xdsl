from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects.irdl.irdl import (
    AttributeOp,
    OperandsOp,
    OperationOp,
    ParametersOp,
    ResultsOp,
)
from xdsl.ir.core import Block, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import (
    AttrConstraint,
    IRDLOperation,
    OpDef,
    ParamAttrDef,
)

from xdsl.dialects.irdl import AnyOp


def constraint_to_irdl(builder: Builder, constraint: AttrConstraint) -> SSAValue:
    """
    Convert an attribute constraint to IRDL.
    This will create new operations at the provided builder location.
    """
    with ImplicitBuilder(builder):
        return AnyOp().output


def op_def_to_irdl(op_def: type[IRDLOperation] | OpDef) -> OperationOp:
    """Convert an operation definition to an IRDL operation definition."""
    if not isinstance(op_def, OpDef):
        op_def = op_def.irdl_definition

    block = Block()
    builder = Builder(block)

    # Operands
    operand_values: list[SSAValue] = []
    for operand in op_def.operands:
        operand_values.append(constraint_to_irdl(builder, operand[1].constr))
    builder.insert(OperandsOp(operand_values))

    # Results
    result_values: list[SSAValue] = []
    for result in op_def.results:
        result_values.append(constraint_to_irdl(builder, result[1].constr))
    builder.insert(ResultsOp(result_values))

    return OperationOp(op_def.name, Region([block]))


def attr_def_to_irdl(
    attr_def: type[ParametrizedAttribute] | ParamAttrDef,
) -> AttributeOp:
    """Convert an attribute definition to an IRDL attribute definition."""
    if not isinstance(attr_def, ParamAttrDef):
        attr_def = attr_def.irdl_definition

    block = Block()
    builder = Builder(block)

    # Parameters
    param_values: list[SSAValue] = []
    for param in attr_def.parameters:
        param_values.append(constraint_to_irdl(builder, param[1]))
    builder.insert(ParametersOp(param_values))

    return AttributeOp(attr_def.name, Region([block]))
