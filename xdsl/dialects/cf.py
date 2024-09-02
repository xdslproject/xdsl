from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IntegerType, StringAttr
from xdsl.ir import Block, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    successor_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class Assert(IRDLOperation):
    name = "cf.assert"
    arg = operand_def(IntegerType(1))
    msg = prop_def(StringAttr)

    def __init__(self, arg: Operation | SSAValue, msg: str | StringAttr):
        if isinstance(msg, str):
            msg = StringAttr(msg)
        super().__init__(
            operands=[arg],
            properties={"msg": msg},
        )


@irdl_op_definition
class Branch(IRDLOperation):
    name = "cf.br"

    arguments = var_operand_def(AnyAttr())
    successor = successor_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, dest: Block, *ops: Operation | SSAValue):
        super().__init__(operands=[[op for op in ops]], successors=[dest])


@irdl_op_definition
class ConditionalBranch(IRDLOperation):
    name = "cf.cond_br"

    cond = operand_def(IntegerType(1))
    then_arguments = var_operand_def(AnyAttr())
    else_arguments = var_operand_def(AnyAttr())

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    then_block = successor_def()
    else_block = successor_def()

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        cond: Operation | SSAValue,
        then_block: Block,
        then_ops: Sequence[Operation | SSAValue],
        else_block: Block,
        else_ops: Sequence[Operation | SSAValue],
    ):
        super().__init__(
            operands=[cond, then_ops, else_ops], successors=[then_block, else_block]
        )


Cf = Dialect(
    "cf",
    [
        Assert,
        Branch,
        ConditionalBranch,
    ],
    [],
)
