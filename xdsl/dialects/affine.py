from __future__ import annotations

from typing import Annotated

from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, IndexType
from xdsl.ir import Operation, SSAValue, Block, Region, Dialect
from xdsl.irdl import (VarOpResult, irdl_op_definition, AttributeDef,
                       RegionDef, VarOperand, AnyAttr)


@irdl_op_definition
class For(Operation):
    name: str = "affine.for"

    arguments: Annotated[VarOperand, AnyAttr()]
    res: Annotated[VarOpResult, AnyAttr()]

    # TODO the bounds are in fact affine_maps
    # TODO support dynamic bounds as soon as maps are here
    lower_bound = AttributeDef(IntegerAttr)
    upper_bound = AttributeDef(IntegerAttr)
    step = AttributeDef(IntegerAttr)

    body = RegionDef()

    def verify_(self) -> None:
        if len(self.operands) != len(self.results):
            raise Exception("Expected the same amount of operands and results")

        operand_types = [SSAValue.get(op).typ for op in self.operands]
        if (operand_types != [res.typ for res in self.results]):
            raise Exception(
                "Expected all operands and result pairs to have matching types"
            )

        entry_block: Block = self.body.blocks[0]
        block_arg_types = [IndexType()] + operand_types
        arg_types = [arg.typ for arg in entry_block.args]
        if block_arg_types != arg_types:
            raise Exception(
                "Expected BlockArguments to have the same types as the operands"
            )

    @staticmethod
    def from_region(operands: list[Operation | SSAValue],
                    lower_bound: int | AnyIntegerAttr,
                    upper_bound: int | AnyIntegerAttr,
                    region: Region,
                    step: int | AnyIntegerAttr = 1) -> For:
        result_types = [SSAValue.get(op).typ for op in operands]
        attributes = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }
        return For.build(operands=[[operand for operand in operands]],
                         result_types=[result_types],
                         attributes=attributes,
                         regions=[region])

    @staticmethod
    def from_callable(operands: list[Operation | SSAValue],
                      lower_bound: int | AnyIntegerAttr,
                      upper_bound: int | AnyIntegerAttr,
                      body: Block.BlockCallback,
                      step: int | AnyIntegerAttr = 1) -> For:
        arg_types = [IndexType()] + [SSAValue.get(op).typ for op in operands]
        return For.from_region(
            operands, lower_bound, upper_bound,
            Region.from_block_list([Block.from_callable(arg_types, body)]),
            step)


@irdl_op_definition
class Yield(Operation):
    name: str = "affine.yield"
    arguments: Annotated[VarOperand, AnyAttr()]

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(
            operands=[SSAValue.get(operand) for operand in operands])


Affine = Dialect([For, Yield], [])
