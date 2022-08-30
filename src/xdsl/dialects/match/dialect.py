from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *


@dataclass
class Match:
    ctx: MLContext

    def __post_init__(self):
        # Types
        self.ctx.register_attr(TypeType)
        self.ctx.register_attr(ValueType)
        self.ctx.register_attr(OperationType)

        # Ops
        self.ctx.register_op(TypeOp)
        self.ctx.register_op(AttributeOp)
        self.ctx.register_op(OperationOp)
        self.ctx.register_op(RootOperationOp)
        self.ctx.register_op(GetResultOp)


class MatchOperation(Operation, ABC):
    pass


@irdl_attr_definition
class TypeType(ParametrizedAttribute):
    name = "type"
    type: ParameterDef[Attribute]


@irdl_attr_definition
class ValueType(ParametrizedAttribute):
    name = "value"


@irdl_attr_definition
class OperationType(ParametrizedAttribute):
    name = "operation"


@irdl_op_definition
class TypeOp(MatchOperation):
    name: str = "match.type"
    output = ResultDef(TypeType)
    type_constraint = OptAttributeDef(StringAttr)


@irdl_op_definition
class AttributeOp(MatchOperation):
    name: str = "match.attr"
    output = ResultDef(Attribute)
    name_constraint = AttributeDef(StringAttr)


@irdl_op_definition
class GetResultOp(MatchOperation):
    name: str = "match.get_result"
    op = OperandDef(OperationType)
    idx = AttributeDef(IntegerAttr)
    output = ResultDef(ValueType)


@irdl_op_definition
class OperationOp(MatchOperation):
    name: str = "match.op"
    operands_constraint = VarOperandDef(ValueType)
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(TypeType)
    name_constraint = OptAttributeDef(StringAttr)
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class RootOperationOp(OperationOp):
    name: str = "match.root_op"
