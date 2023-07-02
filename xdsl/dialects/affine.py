from __future__ import annotations

from typing import Annotated, Sequence

from xdsl.dialects.builtin import AffineMapAttr, AnyIntegerAttr, IndexType, IntegerAttr
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Attribute, Operation, SSAValue, Block, Region, Dialect
from xdsl.traits import IsTerminator
from xdsl.irdl import (
    ConstraintVar,
    VarOpResult,
    attr_def,
    irdl_op_definition,
    VarOperand,
    AnyAttr,
    IRDLOperation,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)


@irdl_op_definition
class For(IRDLOperation):
    name = "affine.for"

    arguments: VarOperand = var_operand_def(AnyAttr())
    res: VarOpResult = var_result_def(AnyAttr())

    # TODO the bounds are in fact affine_maps
    # TODO support dynamic bounds as soon as maps are here
    lower_bound: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    upper_bound: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    step: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    body: Region = region_def()

    # TODO this requires the ImplicitAffineTerminator trait instead of
    # NoTerminator
    # gh issue: https://github.com/xdslproject/xdsl/issues/1149

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
        result_types: Sequence[Attribute],
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
        attributes: dict[str, Attribute] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }
        return For.build(
            operands=[operands],
            result_types=[result_types],
            attributes=attributes,
            regions=[region],
        )


@irdl_op_definition
class Store(IRDLOperation):
    name = "affine.store"

    T = Annotated[Attribute, ConstraintVar("T")]

    value = operand_def(T)
    memref = operand_def(MemRefType[T])
    map = opt_attr_def(AffineMapAttr)

    def __init__(self, value: SSAValue, memref: SSAValue, map: AffineMapAttr):
        super().__init__(
            operands=(value, memref),
            attributes={"map": map},
        )


@irdl_op_definition
class Load(IRDLOperation):
    name = "affine.load"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    indices = var_operand_def(IndexType)

    result = result_def(T)

    map = opt_attr_def(AffineMapAttr)

    def __init__(
        self,
        memref: SSAValue,
        indices: Sequence[SSAValue],
        map: AffineMapAttr,
        result_type: T,
    ):
        super().__init__(
            operands=(memref, indices),
            attributes={"map": map},
            result_types=(result_type,),
        )


@irdl_op_definition
class Yield(IRDLOperation):
    name = "affine.yield"
    arguments: VarOperand = var_operand_def(AnyAttr())

    traits = frozenset([IsTerminator()])

    @staticmethod
    def get(*operands: SSAValue | Operation) -> Yield:
        return Yield.create(operands=[SSAValue.get(operand) for operand in operands])


Affine = Dialect(
    [
        For,
        Store,
        Load,
        Yield,
    ],
    [],
)
