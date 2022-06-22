from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import IntegerType, IntegerAttr


@dataclass
class Tensat:
    """
    This is a dialect purely to enable expression of examples with multi root rewriting. 
    examples using this dialect are inspired by: https://proceedings.mlsys.org/paper/2021/file/65ded5353c5ee48d0b7d48c591b8f430-Paper.pdf
    """
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(MatMul)
        self.ctx.register_op(Concat)
        self.ctx.register_op(Split)


@irdl_op_definition
class MatMul(Operation):
    name: str = "tensat.matmul"
    input1 = OperandDef(AnyAttr())
    input2 = OperandDef(AnyAttr())
    output = ResultDef(AnyAttr())

@irdl_op_definition
class Concat(Operation):
    name: str = "tensat.concat"
    axis = OperandDef(AnyAttr())
    input1 = OperandDef(AnyAttr())
    input2 = OperandDef(AnyAttr())
    output = ResultDef(AnyAttr())

@irdl_op_definition
class Split(Operation):
    name: str = "tensat.split"
    axis = OperandDef(AnyAttr())
    input = OperandDef(AnyAttr())
    output1 = ResultDef(AnyAttr())
    output2 = ResultDef(AnyAttr())

