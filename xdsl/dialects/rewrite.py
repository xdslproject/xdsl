from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *


@dataclass
class Rewrite:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(RewriteId)


@irdl_op_definition
class RewriteId(Operation):
    name: str = "rewrite.id"
    input = OperandDef(Attribute)
    output = ResultDef(Attribute)
