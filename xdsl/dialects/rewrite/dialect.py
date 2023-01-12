from __future__ import annotations
from xdsl.dialects.builtin import IntAttr, StringAttr
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.utils import *
from xdsl.dialects.IRUtils.dialect import AttributeType, OperationType, RangeType, TypeType
from xdsl.dialects.pdl.dialect import OperationOp

##############################################################################
############################ Rewriting interface #############################
##############################################################################


@irdl_op_definition
class ReturnOp(Operation):
    name: str = "rewrite.return"
    result: Annotated[VarOperand, OperationType]


@irdl_op_definition
class ReplaceOperationOp(OperationOp):
    name: str = "rewrite.replace_op"
    body: RegionDef = RegionDef()


# Used in Elevate internals
@irdl_op_definition
class RewriteId(Operation):
    name: str = "rewrite.id"
    input: Annotated[Operand, Attribute]
    output: Annotated[OpResult, Attribute]


Rewrite = Dialect([
    ReturnOp,
    ReplaceOperationOp,
    RewriteId,
], [])