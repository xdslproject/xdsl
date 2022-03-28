from __future__ import annotations
from typing import Union

from xdsl.ir import Operation, SSAValue
from dataclasses import dataclass
from xdsl.dialects.builtin import IntegerAttr, IndexType
from xdsl.irdl import irdl_op_definition, AttributeDef, OperandDef, RegionDef, VarResultDef, VarOperandDef, AnyAttr


@dataclass
class Affine:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(For)
        self.ctx.register_op(Yield)


@irdl_op_definition
class For(Operation):
    name: str = "affine.for"

    arguments = VarOperandDef(AnyAttr())
    res = VarResultDef(AnyAttr())

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
        if ([IndexType()] + operand_types !=
            [arg.typ for arg in entry_block.args]):
            raise Exception(
                "Expected BlockArguments to have the same types as the operands"
            )

    @staticmethod
    def from_region(operands: List[Union[Operation, SSAValue]],
                    lower_bound: Union[int, IntegerAttr],
                    upper_bound: Union[int, IntegerAttr],
                    region: Region,
                    step: Union[int, IntegerAttr] = 1) -> For:
        result_types = [SSAValue.get(op).typ for op in operands]
        return For.build(operands=[[operand for operand in operands]],
                         result_types=[result_types],
                         attributes={
                             "lower_bound": lower_bound,
                             "upper_bound": upper_bound,
                             "step": step
                         },
                         regions=[region])

    @staticmethod
    def from_callable(operands: List[Union[Operation, SSAValue]],
                      lower_bound: Union[int, IntegerAttr],
                      upper_bound: Union[int, IntegerAttr],
                      body: Callable[[BlockArgument, ...], List[Operation]],
                      step: Union[int, IntegerAttr] = 1) -> For:
        arg_types = [IndexType()] + [SSAValue.get(op).typ for op in operands]
        return For.from_region(
            operands, lower_bound, upper_bound,
            Region.from_block_list([Block.from_callable(arg_types, body)]),
            step)


@irdl_op_definition
class Yield(Operation):
    name: str = "affine.yield"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*operands: Union[Operation, SSAValue]) -> Yield:
        return Yield.create(operands=[operand for operand in operands])
