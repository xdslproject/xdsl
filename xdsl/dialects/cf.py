from __future__ import annotations
from collections.abc import Sequence

from typing import Union, Sequence

from xdsl.dialects.builtin import IntegerType, StringAttr
from xdsl.ir import SSAValue, Operation, Block, Dialect
from xdsl.irdl import (
    OpAttr,
    irdl_op_definition,
    VarOperand,
    AnyAttr,
    Operand,
    AttrSizedOperandSegments,
    IRDLOperation,
    Successor,
    operand_def,
    successor_def,
    var_operand_def,
)


@irdl_op_definition
class Assert(IRDLOperation):
    name = "cf.assert"
    arg: Operand = operand_def(IntegerType(1))
    msg: OpAttr[StringAttr]

    @staticmethod
    def get(arg: Operation | SSAValue, msg: str | StringAttr) -> Assert:
        if isinstance(msg, str):
            msg = StringAttr(msg)
        return Assert.build(operands=[arg], attributes={"msg": msg})


@irdl_op_definition
class Branch(IRDLOperation):
    name = "cf.br"

    arguments: VarOperand = var_operand_def(AnyAttr())
    successor: Successor = successor_def()

    @staticmethod
    def get(dest: Block, *ops: Union[Operation, SSAValue]) -> Branch:
        return Branch.build(operands=[[op for op in ops]], successors=[dest])


@irdl_op_definition
class ConditionalBranch(IRDLOperation):
    name = "cf.cond_br"

    cond: Operand = operand_def(IntegerType(1))
    then_arguments: VarOperand = var_operand_def(AnyAttr())
    else_arguments: VarOperand = var_operand_def(AnyAttr())

    irdl_options = [AttrSizedOperandSegments()]

    then_block: Successor = successor_def()
    else_block: Successor = successor_def()

    @staticmethod
    def get(
        cond: Union[Operation, SSAValue],
        then_block: Block,
        then_ops: Sequence[Union[Operation, SSAValue]],
        else_block: Block,
        else_ops: Sequence[Union[Operation, SSAValue]],
    ) -> ConditionalBranch:
        return ConditionalBranch.build(
            operands=[cond, then_ops, else_ops], successors=[then_block, else_block]
        )


Cf = Dialect([Assert, Branch, ConditionalBranch], [])
