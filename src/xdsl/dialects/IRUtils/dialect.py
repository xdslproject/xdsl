from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *


@dataclass
class IRUtils:
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

        # Core Rewriting interface
        self.ctx.register_op(NewOp)
        self.ctx.register_op(FromOp)
        self.ctx.register_op(NewBlockOp)
        self.ctx.register_op(FromBlockOp)
        self.ctx.register_op(RegionFromBlocksOp)
        self.ctx.register_op(NewBlockArgsOp)
        # Ops for rhs
        self.ctx.register_op(GetResultsOp)
        self.ctx.register_op(GetTypeOp)
        self.ctx.register_op(GetAttributes)
        self.ctx.register_op(GetAttribute)
        self.ctx.register_op(NativeMatcherOp)
        self.ctx.register_op(GetNestedOps)
        self.ctx.register_op(GetOperands)

        self.ctx.register_op(GetBlockArgs)
        self.ctx.register_op(GetIndexOfOpInRange)

        # Rewriting Utilities
        self.ctx.register_op(ConcatOp)
        self.ctx.register_op(AddAttributeOp)
        self.ctx.register_op(ArrayAttrElementWiseOp)
        self.ctx.register_op(ApplyNativeRewriteOp)
        self.ctx.register_op(ConstructTypeOp)


##############################################################################
########################## Core Rewriting interface ##########################
##############################################################################


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


##############################################################################
########################## Core Rewriting interface ##########################
##############################################################################


@irdl_op_definition
class NewOp(Operation):
    name: str = "irutils.new_op"
    # TODO: properly specify


@irdl_op_definition
class FromOp(Operation):
    name: str = "irutils.from_op"
    # TODO: properly specify


@irdl_op_definition
class NewBlockOp(Operation):
    name: str = "irutils.new_block"
    # TODO: properly specify


@irdl_op_definition
class FromBlockOp(Operation):
    name: str = "irutils.from_block"
    # TODO: properly specify


@irdl_op_definition
class NewBlockArgsOp(Operation):
    name: str = "irutils.new_block_args"
    types = OperandDef(RangeType)  # range of types
    output = ResultDef(RangeType)  # range of BlockArgs


@irdl_op_definition
class RegionFromBlocksOp(Operation):
    name: str = "irutils.region_from_blocks"
    # TODO: properly specify


##############################################################################
############################ Rewriting Utilities #############################
##############################################################################


@irdl_op_definition
class ConcatOp(Operation):
    name: str = "irutils.concat"
    ranges = VarOperandDef(RangeType)
    output = ResultDef(RangeType)


@irdl_op_definition
class AddAttributeOp(Operation):
    name: str = "irutils.add_attribute"
    ranges = VarOperandDef(RangeType)  # or AttrType
    output = ResultDef(RangeType)


@irdl_op_definition
class ArrayAttrElementWiseOp(Operation):
    name: str = "irutils.array_attr_element_wise"
    array0 = OperandDef(AttributeType)
    array1 = OperandDef(AttributeType)
    output = ResultDef(AttributeType)


@irdl_op_definition
class ApplyNativeRewriteOp(Operation):
    name: str = "irutils.apply_native_rewrite"
    args = VarOperandDef(RangeType)
    rewriter_name = AttributeDef(StringAttr)
    output = ResultDef(Attribute)


@irdl_op_definition
class ConstructTypeOp(Operation):
    name: str = "irutils.construct_type"
    args = VarOperandDef(Attribute)
    output = ResultDef(TypeType)


@irdl_op_definition
class GetResultsOp(Operation):
    name: str = "irutils.get_results"
    op = OperandDef(OperationType)
    idx = OptAttributeDef(IntegerAttr)  # if not specified returns all results
    output = ResultDef(ValueType)  # or RangeType


@irdl_op_definition
class GetTypeOp(Operation):
    """
    Used to get the type of a value or from a range of values
    """
    name: str = "irutils.get_type"
    value = OperandDef(Attribute)  # either ValueType or RangeType
    output = ResultDef(Attribute)  # either TypeType or RangeType


@irdl_op_definition
class NativeMatcherOp(Operation):
    name: str = "irutils.native_matcher"
    matcher_name = AttributeDef(StringAttr)
    output = ResultDef(ValueType)


@irdl_op_definition
class GetNestedOps(Operation):
    """
    Get the ops of a region excluding the terminator.
    """
    name: str = "irutils.get_nested_ops"
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
    name: str = "irutils.get_index_of_op_in_range"
    op = OperandDef(OperationType)
    list = OperandDef(RangeType)
    output = ResultDef(IndexType)


@irdl_op_definition
class GetOperands(Operation):
    name: str = "irutils.get_operands"
    input = OperandDef(OperationType)
    output = ResultDef(RangeType)


@irdl_op_definition
class GetAttributes(Operation):
    name: str = "irutils.get_attributes"
    input = OperandDef(OperationType)
    output = ResultDef(RangeType)


@irdl_op_definition
class GetAttribute(Operation):
    name: str = "irutils.get_attribute"
    input = OperandDef(OperationType)
    output = ResultDef(AttributeType)


@irdl_op_definition
class GetBlockArgs(Operation):
    name: str = "irutils.get_block_args"
    input = OperandDef(OperationType)
    region_idx = OptAttributeDef(IntAttr)
    block_idx = OptAttributeDef(IntAttr)
    output = ResultDef(RangeType)