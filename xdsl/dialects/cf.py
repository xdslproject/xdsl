from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IntegerType, StringAttr
from xdsl.ir import Block, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    successor_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class Assert(IRDLOperation):
    """Assert operation with message attribute"""

    name = "cf.assert"

    arg = operand_def(IntegerType(1))
    msg = attr_def(StringAttr)

    def __init__(self, arg: Operation | SSAValue, msg: str | StringAttr):
        if isinstance(msg, str):
            msg = StringAttr(msg)
        super().__init__(
            operands=[arg],
            attributes={"msg": msg},
        )

    assembly_format = "$arg `,` $msg attr-dict"


@irdl_op_definition
class Branch(IRDLOperation):
    """Branch operation"""

    name = "cf.br"

    arguments = var_operand_def()
    successor = successor_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, dest: Block, *ops: Operation | SSAValue):
        super().__init__(operands=[[op for op in ops]], successors=[dest])

    assembly_format = "$successor (`(` $arguments^ `:` type($arguments) `)`)? attr-dict"


@irdl_op_definition
class ConditionalBranch(IRDLOperation):
    """Conditional branch operation"""

    name = "cf.cond_br"

    cond = operand_def(IntegerType(1))
    then_arguments = var_operand_def()
    else_arguments = var_operand_def()

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

    assembly_format = """
    $cond `,`
    $then_block (`(` $then_arguments^ `:` type($then_arguments) `)`)? `,`
    $else_block (`(` $else_arguments^ `:` type($else_arguments) `)`)?
    attr-dict
    """


Cf = Dialect(
    "cf",
    [
        Assert,
        Branch,
        ConditionalBranch,
    ],
    [],
)
