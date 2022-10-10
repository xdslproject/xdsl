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
        self.ctx.register_op(NewBlock)
        self.ctx.register_op(FromBlock)
        self.ctx.register_op(RegionFromBlocks)
        self.ctx.register_op(NewBlockArgs)
        self.ctx.register_op(ReplaceUses)
        # Ops for rhs
        self.ctx.register_op(GetOp)
        self.ctx.register_op(GetResults)
        self.ctx.register_op(GetType)
        self.ctx.register_op(GetAttributes)
        self.ctx.register_op(GetAttribute)
        self.ctx.register_op(HasAttribute)
        self.ctx.register_op(NativeMatcher)
        self.ctx.register_op(GetNestedOps)
        self.ctx.register_op(GetOperands)
        self.ctx.register_op(GetOperand)

        self.ctx.register_op(GetBlockArgs)
        self.ctx.register_op(GetIndexOfOpInRange)
        self.ctx.register_op(RemoveElement)

        # Rewriting Utilities
        self.ctx.register_op(ForEach)
        self.ctx.register_op(If)
        self.ctx.register_op(Yield)
        self.ctx.register_op(Concat)
        self.ctx.register_op(GetElem)
        self.ctx.register_op(AttributeRange)
        self.ctx.register_op(ArrayAttrElementWise)
        self.ctx.register_op(ApplyNativeRewrite)
        self.ctx.register_op(ConstructType)


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
class NewBlock(Operation):
    name: str = "irutils.new_block"
    # TODO: properly specify


@irdl_op_definition
class FromBlock(Operation):
    name: str = "irutils.from_block"
    # TODO: properly specify


@irdl_op_definition
class RegionFromBlocks(Operation):
    name: str = "irutils.region_from_blocks"
    # TODO: properly specify


@irdl_op_definition
class NewBlockArgs(Operation):
    name: str = "irutils.new_block_args"
    types = OperandDef(RangeType)  # range of types
    output = ResultDef(RangeType)  # range of BlockArgs


@irdl_op_definition
class ReplaceUses(Operation):
    name: str = "irutils.replace_uses"
    old_use = OperandDef(ValueType)
    new_use = OperandDef(ValueType)
    output = ResultDef(RangeType)


##############################################################################
############################ Rewriting Utilities #############################
##############################################################################


@irdl_op_definition
class ForEach(Operation):
    """
    Applies its body to each of the elements of the range. 
    The body has to return a new element for the result range.
    """
    name: str = "irutils.for_each"
    range = OperandDef(RangeType)
    body = RegionDef()
    output = ResultDef(RangeType)


@irdl_op_definition
class If(Operation):
    """
    Applies its body to each of the elements of the range. 
    The body has to return a new element for the result range.
    """
    name: str = "irutils.if"
    condition = OperandDef(i1)
    then_region = RegionDef()
    else_region = RegionDef()


@irdl_op_definition
class Yield(Operation):
    """
    Used to return a value from e.g. ForEach or If
    """
    name: str = "irutils.yield"
    result = OperandDef(Attribute)


@irdl_op_definition
class Concat(Operation):
    name: str = "irutils.concat"
    ranges = VarOperandDef(RangeType)
    output = ResultDef(RangeType)


@irdl_op_definition
class GetElem(Operation):
    """
    returns the element at the given index of a range
    """
    name: str = "irutils.get_elem"
    range = OperandDef(RangeType)
    index = OptOperandDef(Attribute)
    output = ResultDef(Attribute)
    # index is either an Attribute or an Operand


@irdl_op_definition
class AttributeRange(Operation):
    name: str = "irutils.attribute_range"
    ranges = VarOperandDef(RangeType)  # or AttrType
    output = ResultDef(RangeType)


@irdl_op_definition
class ArrayAttrElementWise(Operation):
    name: str = "irutils.array_attr_element_wise"
    array0 = OperandDef(AttributeType)
    array1 = OperandDef(AttributeType)
    output = ResultDef(AttributeType)


@irdl_op_definition
class ApplyNativeRewrite(Operation):
    name: str = "irutils.apply_native_rewrite"
    args = VarOperandDef(RangeType)
    rewriter_name = AttributeDef(StringAttr)
    output = ResultDef(Attribute)


@irdl_op_definition
class ConstructType(Operation):
    name: str = "irutils.construct_type"
    args = VarOperandDef(Attribute)
    output = ResultDef(TypeType)


@irdl_op_definition
class GetOp(Operation):
    """
    Returns the op that created the value
    """
    name: str = "irutils.get_op"
    value = OperandDef(ValueType)
    output = ResultDef(OperationType)  # or RangeType


@irdl_op_definition
class GetResults(Operation):
    name: str = "irutils.get_results"
    op = OperandDef(OperationType)
    index = OptAttributeDef(
        IntegerAttr)  # if not specified returns all results
    output = ResultDef(ValueType)  # or RangeType


@irdl_op_definition
class GetType(Operation):
    """
    Used to get the type of a value or from a range of values
    """
    name: str = "irutils.get_type"
    value = OperandDef(Attribute)  # either ValueType or RangeType
    output = ResultDef(Attribute)  # either TypeType or RangeType


@irdl_op_definition
class NativeMatcher(Operation):
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
    """
    name: str = "irutils.get_index_of_op_in_range"
    op = OperandDef(OperationType)
    list = OperandDef(RangeType)
    output = ResultDef(IndexType)


@irdl_op_definition
class RemoveElement(Operation):
    """
    Remove the element at the given index from the range
    """
    name: str = "irutils.remove_element"
    range = OperandDef(RangeType)
    index = AttributeDef(IntAttr)
    # index can also be given as a second operand


@irdl_op_definition
class GetOperands(Operation):
    name: str = "irutils.get_operands"
    input = OperandDef(OperationType)
    output = ResultDef(RangeType)


@irdl_op_definition
class GetOperand(Operation):
    name: str = "irutils.get_operand"
    input = OperandDef(OperationType)
    index = OptOperandDef(Attribute)
    output = ResultDef(ValueType)
    # index is either specified as an attribute or as an operand


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
class HasAttribute(Operation):
    name: str = "irutils.has_attribute"
    input = OperandDef(AttributeType)
    output = ResultDef(IntegerType)


@irdl_op_definition
class GetBlockArgs(Operation):
    name: str = "irutils.get_block_args"
    input = OperandDef(OperationType)
    region_idx = OptAttributeDef(IntAttr)
    block_idx = OptAttributeDef(IntAttr)
    output = ResultDef(RangeType)