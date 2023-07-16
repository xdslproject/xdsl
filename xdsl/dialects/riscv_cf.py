from abc import ABC
from typing import Sequence

from xdsl.dialects import riscv
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.riscv import AssemblyInstructionArg, RegisterType, RISCVInstruction
from xdsl.ir.core import Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    Successor,
    VarOperand,
    irdl_op_definition,
    operand_def,
    successor_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException


class BranchOperation(IRDLOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V branch operations. Lowers to RsRsOffOperation.
    """

    rs1: Operand = operand_def(RegisterType)
    rs2: Operand = operand_def(RegisterType)

    then_arguments: VarOperand = var_operand_def(RegisterType)
    else_arguments: VarOperand = var_operand_def(RegisterType)

    irdl_options = [AttrSizedOperandSegments()]

    then_block = successor_def()
    else_block = successor_def()

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        then_arguments: Sequence[SSAValue],
        else_arguments: Sequence[SSAValue],
        then_block: Successor,
        else_block: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2, then_arguments, else_arguments],
            attributes={
                "comment": comment,
            },
            successors=(then_block, else_block),
        )

    def verify_(self) -> None:
        # The then block must start with a label op

        then_block_first_op = self.then_block.first_op

        if then_block_first_op is None:
            raise VerifyException("riscv_cf branch op then block must not be empty")

        if not isinstance(then_block_first_op, riscv.LabelOp):
            raise VerifyException(
                "riscv_cf branch op then block first op must be a label"
            )

        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.then_arguments, self.then_block.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        for op_arg, block_arg in zip(self.else_arguments, self.else_block.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        # The else block must be the one immediately following this one

        parent_block = self.parent
        if parent_block is None:
            return

        parent_region = parent_block.parent
        if parent_region is None:
            return

        this_index = parent_region.blocks.index(parent_block)
        else_index = parent_region.blocks.index(self.else_block)

        if this_index + 1 != else_index:
            raise VerifyException(
                "riscv_cf branch op else block must be immediately after op"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        then_label = self.then_block.first_op
        assert isinstance(then_label, riscv.LabelOp)
        return self.rs1, self.rs2, then_label.label


@irdl_op_definition
class BeqOp(BranchOperation):
    """
    Take the branch if registers rs1 and rs2 are equal.

    if (x[rs1] == x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq
    """

    name = "riscv_cf.beq"


@irdl_op_definition
class BneOp(BranchOperation):
    """
    Take the branch if registers rs1 and rs2 are not equal.

    if (x[rs1] != x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bne
    """

    name = "riscv_cf.bne"


@irdl_op_definition
class BltOp(BranchOperation):
    """
    Take the branch if registers rs1 is less than rs2, using signed comparison.

    if (x[rs1] <s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#blt
    """

    name = "riscv_cf.blt"


@irdl_op_definition
class BgeOp(BranchOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using signed comparison.

    if (x[rs1] >=s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bge
    """

    name = "riscv_cf.bge"


@irdl_op_definition
class BltuOp(BranchOperation):
    """
    Take the branch if registers rs1 is less than rs2, using unsigned comparison.

    if (x[rs1] <u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bltu
    """

    name = "riscv_cf.bltu"


@irdl_op_definition
class BgeuOp(BranchOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using unsigned comparison.

    if (x[rs1] >=u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bgeu
    """

    name = "riscv_cf.bgeu"


@irdl_op_definition
class JOp(IRDLOperation, RISCVInstruction):
    """
    A pseudo-instruction, for unconditional jumps you don't expect to return from.
    Is equivalent to JalOp with `rd` = `x0`.
    Used to be a part of the spec, removed in 2.0.
    """

    name = "riscv_cf.j"

    block_arguments: VarOperand = var_operand_def(RegisterType)

    successor = successor_def()

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        block_arguments: Sequence[SSAValue],
        successor: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[block_arguments],
            attributes={
                "comment": comment,
            },
            successors=(successor,),
        )

    def verify_(self) -> None:
        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.block_arguments, self.successor.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        if not isinstance(self.successor.first_op, riscv.LabelOp):
            raise VerifyException(
                "riscv_cf.j operation successor must have a riscv.label operation as a "
                f"first argument, found {self.successor.first_op}"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        dest_label = self.successor.first_op
        assert isinstance(dest_label, riscv.LabelOp)
        return (dest_label.label,)


RISCV_CF = Dialect(
    [
        JOp,
        BeqOp,
        BneOp,
        BltOp,
        BgeOp,
        BltuOp,
        BgeuOp,
    ],
    [],
)
