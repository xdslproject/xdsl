from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *
from xdsl.dialects.IRUtils.dialect import ValueType, TypeType, OperationType, AttributeType


@irdl_attr_definition
class PatternType(ParametrizedAttribute):
    name = "pattern"


@irdl_op_definition
class TypeOp(Operation):
    name: str = "pdl.type"
    output: Annotated[OpResult, TypeType]
    type = OptAttributeDef(StringAttr)


@irdl_op_definition
class AttributeOp(Operation):
    name: str = "pdl.attr"
    output: Annotated[OpResult, Attribute]
    name_constraint = AttributeDef(StringAttr)


@irdl_op_definition
class OperationOp(Operation):
    name: str = "pdl.operation"
    operands_constraint = VarOperandDef(ValueType)
    type_constraint = OptOperandDef(TypeType)
    output: Annotated[OpResult, OperationType]
    name_constraint = OptAttributeDef(StringAttr)
    operands_ordered = OptAttributeDef(IntAttr)
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class OperandOp(Operation):
    name: str = "pdl.operand"
    type_constraint = OptOperandDef(TypeType)
    output: Annotated[OpResult, ValueType]


PDL = Dialect([
    TypeOp,
    AttributeOp,
    OperationOp,
    OperandOp,
], [PatternType])
