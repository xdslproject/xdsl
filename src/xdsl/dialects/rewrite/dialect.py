from __future__ import annotations
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.match.dialect import AttributeType, OperationType, RangeType, TypeType


@dataclass
class Rewrite:
    ctx: MLContext

    def __post_init__(self):
        # Rewriting interface
        self.ctx.register_op(SuccessOp)
        self.ctx.register_op(NewOp)
        self.ctx.register_op(FromOp)
        self.ctx.register_op(NewBlockOp)
        self.ctx.register_op(FromBlockOp)
        self.ctx.register_op(RegionFromBlocksOp)
        self.ctx.register_op(NewBlockArgsOp)

        # Rewriting Utilities
        self.ctx.register_op(ConcatOp)
        self.ctx.register_op(AddAttributeOp)
        self.ctx.register_op(ArrayAttrElementWiseOp)
        self.ctx.register_op(ApplyNativeRewriteOp)
        self.ctx.register_op(ConstructTypeOp)

        # Used in rewriting to return a value of an existing op
        self.ctx.register_op(RewriteId)


##############################################################################
############################ Rewriting interface #############################
##############################################################################


@irdl_op_definition
class SuccessOp(Operation):
    name: str = "rewrite.success"
    result = VarOperandDef(OperationType)


@irdl_op_definition
class NewOp(Operation):
    name: str = "rewrite.new_op"
    # TODO: properly specify


@irdl_op_definition
class FromOp(Operation):
    name: str = "rewrite.from_op"
    # TODO: properly specify


@irdl_op_definition
class NewBlockOp(Operation):
    name: str = "rewrite.new_block"
    # TODO: properly specify


@irdl_op_definition
class FromBlockOp(Operation):
    name: str = "rewrite.from_block"
    # TODO: properly specify


@irdl_op_definition
class NewBlockArgsOp(Operation):
    name: str = "rewrite.new_block_args"
    types = OperandDef(RangeType)  # range of types
    output = ResultDef(RangeType)  # range of BlockArgs


@irdl_op_definition
class RegionFromBlocksOp(Operation):
    name: str = "rewrite.region_from_blocks"
    # TODO: properly specify


##############################################################################
############################ Rewriting Utilities #############################
##############################################################################


@irdl_op_definition
class ConcatOp(Operation):
    name: str = "rewrite.concat"
    ranges = VarOperandDef(RangeType)
    output = ResultDef(RangeType)


@irdl_op_definition
class AddAttributeOp(Operation):
    name: str = "rewrite.add_attribute"
    ranges = VarOperandDef(RangeType)  # or AttrType
    output = ResultDef(RangeType)


@irdl_op_definition
class ArrayAttrElementWiseOp(Operation):
    name: str = "rewrite.array_attr_element_wise"
    array0 = OperandDef(AttributeType)
    array1 = OperandDef(AttributeType)
    output = ResultDef(AttributeType)


@irdl_op_definition
class ApplyNativeRewriteOp(Operation):
    name: str = "rewrite.apply_native_rewrite"
    args = VarOperandDef(RangeType)
    rewriter_name = AttributeDef(StringAttr)
    output = ResultDef(Attribute)


@irdl_op_definition
class ConstructTypeOp(Operation):
    name: str = "rewrite.construct_type"
    args = VarOperandDef(Attribute)
    output = ResultDef(TypeType)


# Used in Elevate internals
@irdl_op_definition
class RewriteId(Operation):
    name: str = "rewrite.id"
    input = OperandDef(Attribute)
    output = ResultDef(Attribute)