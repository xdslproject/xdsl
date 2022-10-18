from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.IRUtils.dialect import ValueType, TypeType, OperationType, AttributeType


@dataclass
class PDL:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(PatternType)
        # Ops for matching
        self.ctx.register_op(TypeOp)
        self.ctx.register_op(AttributeOp)
        self.ctx.register_op(OperationOp)
        self.ctx.register_op(OperandOp)


@irdl_attr_definition
class PatternType(ParametrizedAttribute):
    name = "pattern"


@irdl_op_definition
class TypeOp(Operation):
    name: str = "pdl.type"
    output = ResultDef(TypeType)
    type = OptAttributeDef(StringAttr)


@irdl_op_definition
class AttributeOp(Operation):
    name: str = "pdl.attr"
    output = ResultDef(Attribute)
    name_constraint = AttributeDef(StringAttr)


@irdl_op_definition
class OperationOp(Operation):
    name: str = "pdl.operation"
    operands_constraint = VarOperandDef(ValueType)
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(OperationType)
    name_constraint = OptAttributeDef(StringAttr)
    operands_ordered = OptAttributeDef(IntAttr)
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class OperandOp(Operation):
    name: str = "pdl.operand"
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(ValueType)
