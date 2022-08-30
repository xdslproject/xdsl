from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.match.dialect import OperationType


@dataclass
class Rewrite:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(NewOp)
        self.ctx.register_op(SuccessOp)

        # Used in Elevate internals
        self.ctx.register_op(RewriteId)


@irdl_op_definition
class NewOp(Operation):
    name: str = "rewrite.new_op"
    # input = OperandDef(Attribute)
    # output = ResultDef(Attribute)


@irdl_op_definition
class SuccessOp(Operation):
    name: str = "rewrite.success"
    result = VarOperandDef(OperationType)


# Used in Elevate internals
@irdl_op_definition
class RewriteId(Operation):
    name: str = "rewrite.id"
    input = OperandDef(Attribute)
    output = ResultDef(Attribute)