from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import IntegerType, Float32Type, IntegerAttr, FlatSymbolRefAttr


@dataclass
class Cf:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Branch)
        self.ctx.register_op(ConditionalBranch)


@irdl_op_definition
class Branch(Operation):
    name: str = "cf.br"

    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(block: Block, *ops: Union[Operation, SSAValue]) -> Branch:
        return Branch.build(operands=[[op for op in ops]], successors=[block])


@irdl_op_definition
class ConditionalBranch(Operation):
    name: str = "cf.cond_br"

    then = OperandDef(IntegerType.from_width(1))
    then_arguments = VarOperandDef(AnyAttr())
    else_arguments = VarOperandDef(AnyAttr())

    irdl_options = [AttrSizedOperandSegments()]

    @staticmethod
    def get(cond: Union[Operation, SSAValue], then_block: Block,
            then_ops: List[Union[Operation, SSAValue]], else_block: Block,
            else_ops: List[Union[Operation, SSAValue]]) -> ConditionalBranch:
        return ConditionalBranch.build(operands=[cond, then_ops, else_ops],
                                       successors=[then_block, else_block])
