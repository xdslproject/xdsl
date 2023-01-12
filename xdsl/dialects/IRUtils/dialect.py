from __future__ import annotations
from xdsl.dialects.builtin import *
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *

##############################################################################
########################## Core Rewriting interface ##########################
##############################################################################


@irdl_attr_definition
class TypeType(ParametrizedAttribute):
    name = "type"


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
    types: Annotated[Operand, RangeType]  # range of types
    output: Annotated[OpResult, RangeType]  # range of BlockArgs


@irdl_op_definition
class ReplaceUses(Operation):
    name: str = "irutils.replace_uses"
    old_use: Annotated[Operand, ValueType]
    new_use: Annotated[Operand, ValueType]
    output: Annotated[OpResult, RangeType]


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
    range: Annotated[Operand, RangeType]
    body = RegionDef()
    output: Annotated[OpResult, RangeType]


@irdl_op_definition
class If(Operation):
    """
    Applies its body to each of the elements of the range. 
    The body has to return a new element for the result range.
    """
    name: str = "irutils.if"
    condition: Annotated[Operand, i1]
    then_region = RegionDef()
    else_region = RegionDef()


@irdl_op_definition
class Yield(Operation):
    """
    Used to return a value from e.g. ForEach or If
    """
    name: str = "irutils.yield"
    result: Annotated[Operand, Attribute]


@irdl_op_definition
class Concat(Operation):
    name: str = "irutils.concat"
    ranges = VarOperandDef(RangeType)
    output: Annotated[OpResult, RangeType]


@irdl_op_definition
class GetElem(Operation):
    """
    returns the element at the given index of a range
    """
    name: str = "irutils.get_elem"
    range: Annotated[Operand, RangeType]
    index = OptOperandDef(Attribute)
    output: Annotated[OpResult, Attribute]
    # index is either an Attribute or an Operand


@irdl_op_definition
class AttributeRange(Operation):
    name: str = "irutils.attribute_range"
    ranges = VarOperandDef(RangeType)  # or AttrType
    output: Annotated[OpResult, RangeType]


@irdl_op_definition
class ArrayAttrElementWise(Operation):
    name: str = "irutils.array_attr_element_wise"
    array0: Annotated[Operand, AttributeType]
    array1: Annotated[Operand, AttributeType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class ApplyNativeRewrite(Operation):
    name: str = "irutils.apply_native_rewrite"
    args = VarOperandDef(RangeType)
    rewriter_name = AttributeDef(StringAttr)
    output: Annotated[OpResult, Attribute]


@irdl_op_definition
class ConstructType(Operation):
    name: str = "irutils.construct_type"
    args = VarOperandDef(Attribute)
    output: Annotated[OpResult, TypeType]


@irdl_op_definition
class GetOp(Operation):
    """
    Returns the op that created the value
    """
    name: str = "irutils.get_op"
    value: Annotated[Operand, ValueType]
    output: Annotated[OpResult, OperationType]  # or RangeType


@irdl_op_definition
class GetResults(Operation):
    name: str = "irutils.get_results"
    op: Annotated[Operand, OperationType]
    index = OptAttributeDef(
        IntegerAttr)  # if not specified returns all results
    output: Annotated[OpResult, ValueType]  # or RangeType


@irdl_op_definition
class GetType(Operation):
    """
    Used to get the type of a value or attribute or from a range of such.
    """
    name: str = "irutils.get_type"
    value: Annotated[Operand, Attribute]  # either ValueType or RangeType
    output: Annotated[OpResult, Attribute]  # either TypeType or RangeType


@irdl_op_definition
class NativeMatcher(Operation):
    name: str = "irutils.native_matcher"
    matcher_name = AttributeDef(StringAttr)
    output: Annotated[OpResult, ValueType]


@irdl_op_definition
class GetNestedOps(Operation):
    """
    Get the ops of a region excluding the terminator.
    """
    name: str = "irutils.get_nested_ops"
    input: Annotated[Operand, OperationType]
    region_idx = OptAttributeDef(IntAttr)
    block_idx = OptAttributeDef(IntAttr)
    lb = OptOperandDef(IntAttr)
    ub = OptOperandDef(IntAttr)
    custom_lb = OptAttributeDef(IntAttr)
    custom_ub = OptAttributeDef(IntAttr)
    exclude_terminator = OptAttributeDef(IntAttr)
    output: Annotated[OpResult, RangeType]
    irdl_options = [AttrSizedOperandSegments()]


@irdl_op_definition
class GetIndexOfOpInRange(Operation):
    """
    """
    name: str = "irutils.get_index_of_op_in_range"
    op: Annotated[Operand, OperationType]
    list: Annotated[Operand, RangeType]
    output: Annotated[OpResult, IndexType]


@irdl_op_definition
class RemoveElement(Operation):
    """
    Remove the element at the given index from the range
    """
    name: str = "irutils.remove_element"
    range: Annotated[Operand, RangeType]
    index = AttributeDef(IntAttr)
    # index can also be given as a second operand


@irdl_op_definition
class GetOperands(Operation):
    name: str = "irutils.get_operands"
    input: Annotated[Operand, OperationType]
    output: Annotated[OpResult, RangeType]


@irdl_op_definition
class GetOperand(Operation):
    name: str = "irutils.get_operand"
    input: Annotated[Operand, OperationType]
    index = OptOperandDef(Attribute)
    output: Annotated[OpResult, ValueType]
    # index is either specified as an attribute or as an operand


@irdl_op_definition
class GetAttributes(Operation):
    name: str = "irutils.get_attributes"
    input: Annotated[Operand, OperationType]
    output: Annotated[OpResult, RangeType]


@irdl_op_definition
class GetAttribute(Operation):
    name: str = "irutils.get_attribute"
    input: Annotated[Operand, OperationType]
    output: Annotated[OpResult, AttributeType]


@irdl_op_definition
class HasAttribute(Operation):
    name: str = "irutils.has_attribute"
    input: Annotated[Operand, AttributeType]
    output: Annotated[OpResult, IntegerType]


@irdl_op_definition
class GetIndex(Operation):
    """
    Returns the index of a BlockArg or an OpResult
    """
    name: str = "irutils.get_index"
    value: Annotated[Operand, ValueType]
    output: Annotated[OpResult, IndexType]


@irdl_op_definition
class GetBlockArgs(Operation):
    name: str = "irutils.get_block_args"
    input: Annotated[Operand, OperationType]
    region_idx = OptAttributeDef(IntAttr)
    block_idx = OptAttributeDef(IntAttr)
    output: Annotated[OpResult, RangeType]


# Tensor Specific
@irdl_op_definition
class ConcatTensors(Operation):
    name: str = "irutils.concat_tensors"
    input = VarOperandDef(ValueType)
    new_tensor: Annotated[OpResult, AttributeType]
    new_tensor_type = OptResultDef(TypeType)


IRUtils = Dialect(
    [
        # Core Rewriting interface
        NewOp,
        FromOp,
        NewBlock,
        FromBlock,
        RegionFromBlocks,
        NewBlockArgs,
        ReplaceUses,
        # Ops for rhs
        GetOp,
        GetResults,
        GetType,
        GetAttributes,
        GetAttribute,
        HasAttribute,
        NativeMatcher,
        GetNestedOps,
        GetOperands,
        GetOperand,
        GetBlockArgs,
        GetIndex,
        GetIndexOfOpInRange,
        RemoveElement,

        # Rewriting Utilities
        ForEach,
        If,
        Yield,
        Concat,
        GetElem,
        AttributeRange,
        ArrayAttrElementWise,
        ApplyNativeRewrite,
        ConstructType,

        # Tensor Specific
        ConcatTensors,
    ],
    [
        TypeType,
        AnyType,
        ValueType,
        OperationType,
        AttributeType,
        RegionType,
        BlockType,
        RangeType,
        NativeHandleType,
    ])
