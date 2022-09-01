from __future__ import annotations
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.match.dialect import OperationType, RangeType


@dataclass
class Rewrite:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(NewOp)
        self.ctx.register_op(FromOp)
        self.ctx.register_op(SuccessOp)

        # Used in Elevate internals
        self.ctx.register_op(RewriteId)


@irdl_op_definition
class NewOp(Operation):
    name: str = "rewrite.new_op"
    # TODO: properly specify


@irdl_op_definition
class FromOp(Operation):
    name: str = "rewrite.from_op"
    # TODO: properly specify


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