from __future__ import annotations
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.IRUtils.dialect import AttributeType, OperationType, RangeType, TypeType
from xdsl.dialects.pdl.dialect import OperationOp


@dataclass
class Rewrite:
    ctx: MLContext

    def __post_init__(self):
        # Rewriting interface
        self.ctx.register_op(ReturnOp)
        self.ctx.register_op(ReplaceOperationOp)

        # Used in rewriting to return a value of an existing op
        self.ctx.register_op(RewriteId)


##############################################################################
############################ Rewriting interface #############################
##############################################################################


@irdl_op_definition
class ReturnOp(Operation):
    name: str = "rewrite.return"
    result = VarOperandDef(OperationType)


@irdl_op_definition
class ReplaceOperationOp(OperationOp):
    name: str = "rewrite.replace_op"
    body: RegionDef = RegionDef()


# Used in Elevate internals
@irdl_op_definition
class RewriteId(Operation):
    name: str = "rewrite.id"
    input = OperandDef(Attribute)
    output = ResultDef(Attribute)