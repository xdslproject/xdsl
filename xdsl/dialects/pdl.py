from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *


@irdl_attr_definition
class AttributeType(ParametrizedAttribute):
    name = "attribute"


@irdl_attr_definition
class OperationType(ParametrizedAttribute):
    name = "operation"


@irdl_attr_definition
class RangeType(ParametrizedAttribute):
    name = "range"


@irdl_attr_definition
class TypeType(ParametrizedAttribute):
    name = "type"


@irdl_attr_definition
class ValueType(ParametrizedAttribute):
    name = "value"


@irdl_op_definition
class AttributeOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
    """
    name: str = "pdl.attribute"
    value: OptOpAttr[Attribute]
    value_type: Annotated[OptOperand, TypeType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class EraseOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlerase-mlirpdleraseop
    """
    name: str = "pdl.erase"
    op_value: Annotated[Operand, OperationType]


@irdl_op_definition
class OperandOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperand-mlirpdloperandop
    """
    name: str = "pdl.operand"
    value_type: Annotated[Operand, TypeType]
    output: Annotated[OpResult, ValueType]


@irdl_op_definition
class OperandsOp(Operation):
    """
    https://mlir.llvm.org/docs/Dialects/PDLOps/#pdloperands-mlirpdloperandsop
    """
    name: str = "pdl.operands"
    value_type: Annotated[Operand,
                          RangeType]  # Range of Types can we parametrize this?
    output: Annotated[OpResult,
                      RangeType]  # Range of Values can we parametrize this?


Pdl = Dialect([AttributeOp, OperandOp, EraseOp, OperandsOp],
              [AttributeType, OperationType, RangeType, TypeType, ValueType])
