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
        self.ctx.register_attr(AnyType)
        self.ctx.register_attr(ValueType)
        self.ctx.register_attr(OperationType)
        self.ctx.register_attr(AttributeType)
        self.ctx.register_attr(RegionType)
        self.ctx.register_attr(BlockType)
        self.ctx.register_attr(RangeType)
        self.ctx.register_attr(NativeHandleType)

        # Ops for matching
        self.ctx.register_op(TypeOp)
        self.ctx.register_op(AttributeOp)
        self.ctx.register_op(OperationOp)
        self.ctx.register_op(RootOperationOp)
        self.ctx.register_op(OperandOp)

        # Ops for rhs
        self.ctx.register_op(GetResultOp)
        self.ctx.register_op(GetTypeOp)
        self.ctx.register_op(GetAttributes)
        self.ctx.register_op(NativeMatcherOp)
        self.ctx.register_op(GetNestedOps)
        self.ctx.register_op(GetOperands)

        self.ctx.register_op(GetBlockArgs)
        self.ctx.register_op(GetIndexOfOpInRange)


class MatchOperation(Operation, ABC):
    pass


@irdl_attr_definition
class TypeType(ParametrizedAttribute):
    name = "type"
    type: ParameterDef[Attribute]


@irdl_attr_definition
class AnyType(ParametrizedAttribute):
    """
    Used as container in TypeType when the type will only be known during interpretation
    """
    name = "any"


@irdl_attr_definition
class ValueType(ParametrizedAttribute):
    name = "value"


@irdl_attr_definition
class OperationType(ParametrizedAttribute):
    name = "operation"


@irdl_attr_definition
class AttributeType(ParametrizedAttribute):
    name = "attribute"


@irdl_attr_definition
class RegionType(ParametrizedAttribute):
    name = "region"


@irdl_attr_definition
class BlockType(ParametrizedAttribute):
    name = "block"


@irdl_attr_definition
class RangeType(ParametrizedAttribute):
    name = "range"
    type: ParameterDef[Attribute]


@irdl_attr_definition
class NativeHandleType(ParametrizedAttribute):
    name = "native_handle"


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
class GetTypeOp(MatchOperation):
    """
    Used to get the type of a value or from a range of values
    """
    name: str = "match.get_type"
    value = OperandDef(Attribute)  # either ValueType or RangeType
    output = ResultDef(Attribute)  # either TypeType or RangeType


@irdl_op_definition
class OperationOp(MatchOperation):
    name: str = "match.op"
    operands_constraint = VarOperandDef(ValueType)
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(OperationType)
    name_constraint = OptAttributeDef(StringAttr)
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class RootOperationOp(OperationOp):
    name: str = "match.root_op"


@irdl_op_definition
class OperandOp(MatchOperation):
    name: str = "match.operand"
    type_constraint = OptOperandDef(TypeType)
    output = ResultDef(ValueType)


@irdl_op_definition
class NativeMatcherOp(MatchOperation):
    name: str = "match.native_matcher"
    matcher_name = AttributeDef(StringAttr)
    output = ResultDef(ValueType)


@irdl_op_definition
class GetNestedOps(Operation):
    """
    Get the ops of a region excluding the terminator.
    """
    name: str = "match.get_nested_ops"
    input = OperandDef(OperationType)
    region_idx = OptAttributeDef(IntAttr)
    block_idx = OptAttributeDef(IntAttr)
    lb = OptOperandDef(IntAttr)
    ub = OptOperandDef(IntAttr)
    custom_lb = OptAttributeDef(IntAttr)
    custom_ub = OptAttributeDef(IntAttr)
    exclude_terminator = OptAttributeDef(IntAttr)
    output = ResultDef(RangeType)
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class GetIndexOfOpInRange(Operation):
    """
    Get the ops of a region excluding the terminator.
    """
    name: str = "match.get_index_of_op_in_range"
    op = OperandDef(OperationType)
    list = OperandDef(RangeType)
    output = ResultDef(IndexType)


@irdl_op_definition
class GetOperands(Operation):
    name: str = "match.get_operands"
    input = OperandDef(OperationType)
    output = ResultDef(RangeType)


@irdl_op_definition
class GetAttributes(Operation):
    name: str = "match.get_attributes"
    input = OperandDef(OperationType)
    output = ResultDef(RangeType)


@irdl_op_definition
class GetBlockArgs(Operation):
    name: str = "match.get_block_args"
    input = OperandDef(OperationType)
    region_idx = OptAttributeDef(IntAttr)
    block_idx = OptAttributeDef(IntAttr)
    output = ResultDef(RangeType)