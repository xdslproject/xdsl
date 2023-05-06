from __future__ import annotations

from typing import Annotated, Sequence

from xdsl.dialects.builtin import AnyIntegerAttr, IndexType, IntegerAttr
from xdsl.ir import Attribute, Operation, SSAValue, Block, Region, Dialect
from xdsl.irdl import (
    OpAttr,
    VarOpResult,
    irdl_op_definition,
    VarOperand,
    AnyAttr,
    IRDLOperation,
)


@irdl_op_definition
class For(IRDLOperation):
    name = "affine.for"

    arguments: Annotated[VarOperand, AnyAttr()]
    res: Annotated[VarOpResult, AnyAttr()]

    # TODO the bounds are in fact affine_maps
    # TODO support dynamic bounds as soon as maps are here
    lower_bound: OpAttr[AnyIntegerAttr]
    upper_bound: OpAttr[AnyIntegerAttr]
    step: OpAttr[AnyIntegerAttr]

    body: Region

    def verify_(self) -> None:
        if len(self.operands) != len(self.results):
            raise Exception("Expected the same amount of operands and results")

        operand_types = [SSAValue.get(op).typ for op in self.operands]
        if operand_types != [res.typ for res in self.results]:
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
    def from_region(
        operands: Sequence[Operation | SSAValue],
        lower_bound: int | AnyIntegerAttr,
        upper_bound: int | AnyIntegerAttr,
        region: Region,
        step: int | AnyIntegerAttr = 1,
    ) -> For:
        if isinstance(lower_bound, int):
            lower_bound = IntegerAttr.from_index_int_value(lower_bound)
        if isinstance(upper_bound, int):
            upper_bound = IntegerAttr.from_index_int_value(upper_bound)
        if isinstance(step, int):
            step = IntegerAttr.from_index_int_value(step)
        result_types = [SSAValue.get(op).typ for op in operands]
        attributes: dict[str, Attribute] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }
        return For.build(
            operands=[[operand for operand in operands]],
            result_types=[result_types],
            attributes=attributes,
            regions=[region],
        )

    @staticmethod
    def from_callable(
        operands: Sequence[Operation | SSAValue],
        lower_bound: int | AnyIntegerAttr,
        upper_bound: int | AnyIntegerAttr,
        body: Block.BlockCallback,
        step: int | AnyIntegerAttr = 1,
    ) -> For:
        arg_types = [IndexType()] + [SSAValue.get(op).typ for op in operands]
        return For.from_region(
            operands,
            lower_bound,
            upper_bound,
            Region(Block.from_callable(arg_types, body)),
            step,
        )


@irdl_op_definition
class Yield(IRDLOperation):
    name = "affine.yield"
    arguments: Annotated[VarOperand, AnyAttr()]

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(operands=[SSAValue.get(operand) for operand in operands])


Affine = Dialect([For, Yield], [])
