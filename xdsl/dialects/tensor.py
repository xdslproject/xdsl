from dataclasses import dataclass
from typing import List, Optional, Sequence
from xdsl.dialects.builtin import ContainerOf, IndexType, IntegerAttr, TensorType, UnrankedTensorType
from xdsl.ir import Attribute, Block, MLContext, Operation, Region, SSAValue
from xdsl.irdl import AnyAttr, AnyOf, OperandDef, RegionDef, ResultDef, VarOperandDef, irdl_op_definition


@dataclass
class Tensor:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Extract)
        self.ctx.register_op(Insert)
        self.ctx.register_op(Yield)


@irdl_op_definition
class Extract(Operation):
    name = "tensor.extract"
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
    def get(tensor: SSAValue | Operation,
            *indices: SSAValue | Operation) -> 'Extract':
        operands = [tensor] + [SSAValue.get(i) for i in indices]
        return Extract.create(operands, result_types=[SSAValue.get(tensor).typ.element_type])


@irdl_op_definition
class Insert(Operation):
    name = "tensor.insert"
    value = OperandDef(AnyAttr())
    tensor = OperandDef(TensorType)
    indices = VarOperandDef(IndexType)

    res = ResultDef(TensorType)

    def verify_(self):
        if self.tensor.typ.element_type != self.value.typ:
            raise Exception(
                "expected value type to match the tensor element type")

        if self.tensor.typ.get_num_dims() != len(self.indices):
            raise Exception("expected an index for each dimension")

    @staticmethod
    def get(value: Operation | SSAValue, tensor: Operation | SSAValue,
            *indices: Operation | SSAValue) -> 'Insert':
        operands = [value, tensor] + [SSAValue.get(i) for i in indices]
        return Insert.create(operands, result_types=[tensor.typ])


@irdl_op_definition
class Yield(Operation):
    name: str = "tensor.yield"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*operands: SSAValue | Operation) -> 'Yield':
        return Yield.create(
            operands=[SSAValue.get(operand) for operand in operands])
