from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.IRUtils.dialect import ValueType, TypeType, OperationType


@dataclass
class Match:
    ctx: MLContext

    def __post_init__(self):
        # Ops for matching
        self.ctx.register_op(TypeOp)
        self.ctx.register_op(AttributeOp)
        self.ctx.register_op(OperationOp)
        self.ctx.register_op(OperandOp)


@irdl_op_definition
class TypeOp(Operation):
    name: str = "match.type"
    output = ResultDef(TypeType)
    type_constraint = OptAttributeDef(StringAttr)


@irdl_op_definition
class AttributeOp(Operation):
    name: str = "match.attr"
    output = ResultDef(Attribute)
    name_constraint = AttributeDef(StringAttr)


@irdl_op_definition
class OperationOp(Operation):
    name: str = "match.op"
    operands_constraint = VarOperandDef(ValueType)
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(OperationType)
    name_constraint = OptAttributeDef(StringAttr)
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class OperandOp(Operation):
    name: str = "match.operand"
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(ValueType)