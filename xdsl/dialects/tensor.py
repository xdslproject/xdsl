from dataclasses import dataclass
from typing import List
from xdsl.dialects.builtin import IndexType, TensorType
from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.irdl import AnyAttr, OperandDef, ResultDef, VarOperandDef, irdl_op_definition


@dataclass
class Tensor:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Insert)


@irdl_op_definition
class Insert(Operation):
    name = "tensor.insert"
    tensor = OperandDef(TensorType)
    indices = VarOperandDef(IndexType)
    res = ResultDef(AnyAttr())

    def verify_(self):
        if self.tensor.typ.element_type != self.res.typ:
            raise Exception(
                "expected return type to match the tensor element type")

        if self.tensor.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @staticmethod
    def get(ref: SSAValue | Operation,
            indices: List[SSAValue | Operation]) -> 'Insert':
        return Insert.build(operands=[ref, indices],
                          result_types=[SSAValue.get(ref).typ.element_type])
