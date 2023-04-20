from abc import ABC
from typing import Sequence
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.riscv import AssemblyInstructionArg, RISCVInstruction, RegisterType
from xdsl.ir.core import Block, Dialect, Operation, SSAValue
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


def instruction_count(block: Block) -> int:
    return sum(isinstance(op, RISCVInstruction) for op in block.ops)


def instruction_offset(source_block: Block, target_block: Block) -> int:
    """
    Offset in number of RISC-V unstructions to jump to go from the last operation in
    `source_block` to the first operation in `target_block`.

    To go from the last operation of one block, to the first operation in the one
    immediately following is 1.
    """

    region = source_block.parent
    assert region is not None
    assert region is target_block.parent

    index_by_block = {block: index for index, block in enumerate(region.blocks)}
    instruction_count_by_block = {
        block: instruction_count(block) for block in region.blocks
    }

    source_index = index_by_block[source_block]
    target_index = index_by_block[target_block]
    if source_index < target_index:
        # Target block is later in the code, add 1 to account for start of next block, and
        # add number of instructions in the blocks in between the two.
        return 1 + sum(
            instruction_count_by_block[region.blocks[index]]
            for index in range(source_index + 1, target_index)
        )
    elif source_index == target_index:
        # Jump to the start of current block, so number of instructions in block - 1
        return 1 - instruction_count_by_block[source_block]
    else:
        # Target block is before in the code, so we jump back by the number of
        # instructions in this block, and all the blocks in between, including target
        # block, -1.
        return 1 - sum(
            instruction_count_by_block[region.blocks[index]]
            for index in range(target_index, source_index + 1)
        )


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

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        block = self.parent
        assert block is not None
        return self.rs1, self.rs2, instruction_offset(block, self.then_block)


@irdl_op_definition
class BeqOp(BranchOperation):
    """
    Take the branch if registers rs1 and rs2 are equal.

    if (x[rs1] == x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq
    """

    name = "riscv.cf.beq"


@irdl_op_definition
class BneOp(BranchOperation):
    """
    Take the branch if registers rs1 and rs2 are not equal.

    if (x[rs1] != x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bne
    """

    name = "riscv.cf.bne"


@irdl_op_definition
class BltOp(BranchOperation):
    """
    Take the branch if registers rs1 is less than rs2, using signed comparison.

    if (x[rs1] <s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#blt
    """

    name = "riscv.cf.blt"


@irdl_op_definition
class BgeOp(BranchOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using signed comparison.

    if (x[rs1] >=s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bge
    """

    name = "riscv.cf.bge"


@irdl_op_definition
class BltuOp(BranchOperation):
    """
    Take the branch if registers rs1 is less than rs2, using unsigned comparison.

    if (x[rs1] <u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bltu
    """

    name = "riscv.cf.bltu"


@irdl_op_definition
class BgeuOp(BranchOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using unsigned comparison.

    if (x[rs1] >=u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bgeu
    """

    name = "riscv.cf.bgeu"


@irdl_op_definition
class JOp(IRDLOperation, RISCVInstruction):
    """
    A pseudo-instruction, for unconditional jumps you don't expect to return from.
    Is equivalent to JalOp with `rd` = `x0`.
    Used to be a part of the spec, removed in 2.0.
    """

    name = "riscv.cf.j"

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

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        block = self.parent
        assert block is not None
        return (instruction_offset(block, self.successor),)


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
