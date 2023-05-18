"""Definition of the IRDL dialect."""

from typing import Annotated
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    Region,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    OpAttr,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
)
from xdsl.traits import HasParent

################################################################################
# Dialect, Operation, and Attribute definitions                                #
################################################################################


@irdl_attr_definition
class AttributeType(ParametrizedAttribute, TypeAttribute):
    """Type of a attribute handle."""

    name = "irdl.attribute"


@irdl_op_definition
class DialectOp(IRDLOperation):
    """A dialect definition."""

    name = "irdl.dialect"

    sym_name: OpAttr[StringAttr]
    body: Region


@irdl_op_definition
class TypeOp(IRDLOperation):
    """A type definition."""

    name = "irdl.type"

    sym_name: OpAttr[StringAttr]
    body: Region

    traits = frozenset([HasParent(DialectOp)])


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """An attribute definition."""

    name = "irdl.attribute"

    sym_name: OpAttr[StringAttr]
    body: Region

    traits = frozenset([HasParent(DialectOp)])


@irdl_op_definition
class ParametersOp(IRDLOperation):
    """An attribute or type parameter definition"""

    name = "irdl.parameters"

    args: Annotated[VarOperand, AttributeType]

    traits = frozenset([HasParent((TypeOp, AttributeOp))])


@irdl_op_definition
class OperationOp(IRDLOperation):
    """An operation definition."""

    name = "irdl.operation"

    sym_name: OpAttr[StringAttr]
    body: Region

    traits = frozenset([HasParent(DialectOp)])


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """An operation operand definition."""

    name = "irdl.operands"

    args: Annotated[VarOperand, AttributeType]

    traits = frozenset([HasParent(OperationOp)])


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """An operation result definition."""

    name = "irdl.results"

    args: Annotated[VarOperand, AttributeType]

    traits = frozenset([HasParent(OperationOp)])


################################################################################
# Attribute constraints                                                        #
################################################################################


@irdl_op_definition
class IsOp(IRDLOperation):
    """Constraint an attribute/type to be a specific attribute instance."""

    name = "irdl.is"

    expected: OpAttr[Attribute]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class ParametricOp(IRDLOperation):
    """Constraint an attribute/type base and its parameters"""

    name = "irdl.parametric"

    base_type: OpAttr[SymbolRefAttr]
    args: Annotated[VarOperand, AttributeType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class AnyOp(IRDLOperation):
    """Constraint an attribute/type to be any attribute/type instance."""

    name = "irdl.any"

    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class AnyOfOp(IRDLOperation):
    """Constraint an attribute/type to the union of the provided constraints."""

    name = "irdl.any_of"

    args: Annotated[VarOperand, AttributeType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class AllOfOp(IRDLOperation):
    """Constraint an attribute/type to the intersection of the provided constraints."""

    name = "irdl.all_of"

    args: Annotated[VarOperand, AttributeType]
    output: Annotated[OpResult, AttributeType]


IRDL = Dialect(
    [
        DialectOp,
        TypeOp,
        AttributeOp,
        ParametersOp,
        OperationOp,
        OperandsOp,
        ResultsOp,
        IsOp,
        ParametricOp,
        AnyOp,
        AnyOfOp,
        AllOfOp,
    ],
    [AttributeType],
)
