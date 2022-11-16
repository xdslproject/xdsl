from dataclasses import dataclass
from typing import List
from xdsl.dialects.builtin import ContainerOf, IndexType, TensorType, UnrankedTensorType
from xdsl.ir import Block, MLContext, Operation, SSAValue
from xdsl.irdl import AnyAttr, AnyOf, OperandDef, RegionDef, ResultDef, VarOperandDef, irdl_op_definition


@dataclass
class Tensor:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Generate)
        self.ctx.register_op(Extract)
        self.ctx.register_op(Yield)


@irdl_op_definition
class Generate(Operation):
    name: str = "tensor.generate"

    indices = VarOperandDef(IndexType)
    res = ResultDef(TensorType)

    body = RegionDef()

    def verify_(self) -> None:
        if self.res.get_num_dims() != len(self.indices):
            raise Exception("Expected the same amount of indices and tensor dimensions")

        # TODO: all operands must be IndexType. Also, anything else missing here?
        operand_types = [SSAValue.get(op).typ for op in self.operands]

        entry_block: Block = self.body.blocks[0]
        block_arg_types = [arg.typ for arg in entry_block.args]
        if block_arg_types != operand_types:
            raise Exception(
                "Expected BlockArguments to have the same types as the operands"
            )


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
        operands=[tensor] + [SSAValue.get(operand) for operand in indices]
        return Extract.create(operands, result_types=[SSAValue.get(tensor).typ.element_type])

@irdl_op_definition
class Yield(Operation):
    name: str = "tensor.yield"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*operands: SSAValue | Operation) -> 'Yield':
        return Yield.create(
            operands=[SSAValue.get(operand) for operand in operands])
