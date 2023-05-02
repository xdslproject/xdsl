from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from xdsl.ir import (
    Dialect,
    Operation,
    SSAValue,
    Data,
    OpResult,
    TypeAttribute,
)

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    irdl_attr_definition,
    Operand,
    OpAttr,
)

from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.dialects.builtin import AnyIntegerAttr


@dataclass(frozen=True)
class Register:
    """
    A RISC-V register.
    """


@irdl_attr_definition
class RegisterType(Data[Register], TypeAttribute):
    """
    A RISC-V register type.
    """

    name = "riscv.reg"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> Register:
        return Register()

    def print_parameter(self, printer: Printer) -> None:
        pass


class RdRsRsOperation(IRDLOperation, ABC):
    """
    A base class for RISC-V operations that have one destination register, and two source
    registers.

    This is called R-Type in the RISC-V specification.
    """

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
    ):
        rd = RegisterType(Register())

        super().__init__(
            operands=[rs1, rs2],
            result_types=[rd],
        )


class RdImmOperation(IRDLOperation, ABC):
    """
    A base class for RISC-V operations that have one destination register, and one
    immediate operand (e.g. U-Type and J-Type instructions in the RISC-V spec).
    """

    rd: Annotated[OpResult, RegisterType]
    immediate: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        immediate: AnyIntegerAttr,
    ):
        rd = RegisterType(Register())
        super().__init__(
            attributes={
                "immediate": immediate,
            },
            result_types=[rd],
        )


class RdRsImmOperation(IRDLOperation, ABC):
    """
    A base class for RISC-V operations that have one destination register, one source
    register and one immediate operand.

    This is called I-Type in the RISC-V specification.
    """

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    immediate: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: AnyIntegerAttr,
    ):
        rd = RegisterType(Register())
        super().__init__(
            operands=[rs1],
            attributes={
                "immediate": immediate,
            },
            result_types=[rd],
        )


class RsRsOffOperation(IRDLOperation, ABC):
    """
    A base class for RISC-V operations that have one source register and a destination
    register, and an offset.

    This is called B-Type in the RISC-V specification.
    """

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    offset: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        offset: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "offset": offset,
            },
        )


class RsRsImmOperation(IRDLOperation, ABC):
    """
    A base class for RISC-V operations that have two source registers and an
    immediate.

    This is called S-Type in the RISC-V specification.
    """

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    immediate: OpAttr[AnyIntegerAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: AnyIntegerAttr,
    ):
        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "immediate": immediate,
            },
        )


class NullaryOperation(IRDLOperation, ABC):
    """
    A base class for RISC-V operations that have neither sources nor destinations.
    """

    pass


# RV32I/RV64I: Integer Computational Instructions (Section 2.4)

## Integer Register-Immediate Instructions


@irdl_op_definition
class AddiOp(RdRsImmOperation):
    """
    Adds the sign-extended 12-bit immediate to register rs1.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#addi
    """

    name = "riscv.addi"


@irdl_op_definition
class SltiOp(RdRsImmOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the sign-extended
    immediate when both are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slti
    """

    name = "riscv.slti"


@irdl_op_definition
class SltiuOp(RdRsImmOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the immediate when
    both are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltiu
    """

    name = "riscv.sltiu"


@irdl_op_definition
class AndiOp(RdRsImmOperation):
    """
    Performs bitwise AND on register rs1 and the sign-extended 12-bit
    immediate and place the result in rd.

    x[rd] = x[rs1] & sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#andi
    """

    name = "riscv.andi"


@irdl_op_definition
class OriOp(RdRsImmOperation):
    """
    Performs bitwise OR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] | sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#ori
    """

    name = "riscv.ori"


@irdl_op_definition
class XoriOp(RdRsImmOperation):
    """
    Performs bitwise XOR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] ^ sext(immediate)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xori
    """

    name = "riscv.xori"


@irdl_op_definition
class SlliOp(RdRsImmOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] << shamt

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli
    """

    name = "riscv.slli"


@irdl_op_definition
class SrliOp(RdRsImmOperation):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>u shamt

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli
    """

    name = "riscv.srli"


@irdl_op_definition
class SraiOp(RdRsImmOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai
    """

    name = "riscv.srai"


@irdl_op_definition
class LuiOp(RdImmOperation):
    """
    Build 32-bit constants and uses the U-type format. LUI places the U-immediate value
    in the top 20 bits of the destination register rd, filling in the lowest 12 bits with zeros.

    x[rd] = sext(immediate[31:12] << 12)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lui
    """

    name = "riscv.lui"


@irdl_op_definition
class AuipcOp(RdImmOperation):
    """
    Build pc-relative addresses and uses the U-type format. AUIPC forms a 32-bit offset
    from the 20-bit U-immediate, filling in the lowest 12 bits with zeros, adds this
    offset to the pc, then places the result in register rd.

    x[rd] = pc + sext(immediate[31:12] << 12)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#auipc
    """

    name = "riscv.auipc"


## Integer Register-Register Operations


@irdl_op_definition
class AddOp(RdRsRsOperation):
    """
    Adds the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#add
    """

    name = "riscv.add"


@irdl_op_definition
class SltOp(RdRsRsOperation):
    """
    Place the value 1 in register rd if register rs1 is less than register rs2 when both
    are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slt
    """

    name = "riscv.slt"


@irdl_op_definition
class SltuOp(RdRsRsOperation):
    """
    Place the value 1 in register rd if register rs1 is less than register rs2 when both
    are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltu
    """

    name = "riscv.sltu"


@irdl_op_definition
class AndOp(RdRsRsOperation):
    """
    Performs bitwise AND on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] & x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#and
    """

    name = "riscv.and"


@irdl_op_definition
class OrOp(RdRsRsOperation):
    """
    Performs bitwise OR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] | x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#or
    """

    name = "riscv.or"


@irdl_op_definition
class XorOp(RdRsRsOperation):
    """
    Performs bitwise XOR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] ^ x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xor
    """

    name = "riscv.xor"


@irdl_op_definition
class SllOp(RdRsRsOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of register rs2.

    x[rd] = x[rs1] << x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sll
    """

    name = "riscv.sll"


@irdl_op_definition
class SrlOp(RdRsRsOperation):
    """
    Logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2.

    x[rd] = x[rs1] >>u x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srl
    """

    name = "riscv.srl"


@irdl_op_definition
class SubOp(RdRsRsOperation):
    """
    Subtracts the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] - x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub
    """

    name = "riscv.sub"


@irdl_op_definition
class SraOp(RdRsRsOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2.

    x[rd] = x[rs1] >>s x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub
    """

    name = "riscv.sra"


@irdl_op_definition
class NopOp(NullaryOperation):
    """
    Does not change any user-visible state, except for advancing the pc register.
    Canonical nop is encoded as addi x0, x0, 0.
    """

    name = "riscv.nop"


# Conditional Branches


@irdl_op_definition
class BeqOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 and rs2 are equal.

    if (x[rs1] == x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq
    """

    name = "riscv.beq"


@irdl_op_definition
class BneOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 and rs2 are not equal.

    if (x[rs1] != x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bne
    """

    name = "riscv.bne"


@irdl_op_definition
class BltOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is less than rs2, using signed comparison.

    if (x[rs1] <s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#blt
    """

    name = "riscv.blt"


@irdl_op_definition
class BgeOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using signed comparison.

    if (x[rs1] >=s x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bge
    """

    name = "riscv.bge"


@irdl_op_definition
class BltuOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is less than rs2, using unsigned comparison.

    if (x[rs1] <u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bltu
    """

    name = "riscv.bltu"


@irdl_op_definition
class BgeuOp(RsRsOffOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using unsigned comparison.

    if (x[rs1] >=u x[rs2]) pc += sext(offset)

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bgeu
    """

    name = "riscv.bgeu"


# RV32I/RV64I: 2.6 Load and Store Instructions


@irdl_op_definition
class LbOp(RdRsImmOperation):
    """
    Loads a 8-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = sext(M[x[rs1] + sext(offset)][7:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lb
    """

    name = "riscv.lb"


@irdl_op_definition
class LbuOp(RdRsImmOperation):
    """
    Loads a 8-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = M[x[rs1] + sext(offset)][7:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lbu
    """

    name = "riscv.lbu"


@irdl_op_definition
class LhOp(RdRsImmOperation):
    """
    Loads a 16-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = sext(M[x[rs1] + sext(offset)][15:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lh
    """

    name = "riscv.lh"


@irdl_op_definition
class LhuOp(RdRsImmOperation):
    """
    Loads a 16-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = M[x[rs1] + sext(offset)][15:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lhu
    """

    name = "riscv.lhu"


@irdl_op_definition
class LwOp(RdRsImmOperation):
    """
    Loads a 32-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    x[rd] = sext(M[x[rs1] + sext(offset)][31:0])

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lw
    """

    name = "riscv.lw"


@irdl_op_definition
class SbOp(RsRsImmOperation):
    """
    Store 8-bit, values from the low bits of register rs2 to memory.

    M[x[rs1] + sext(offset)] = x[rs2][7:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sb
    """

    name = "riscv.sb"


@irdl_op_definition
class ShOp(RsRsImmOperation):
    """
    Store 16-bit, values from the low bits of register rs2 to memory.

    M[x[rs1] + sext(offset)] = x[rs2][15:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sh

    """

    name = "riscv.sh"


@irdl_op_definition
class SwOp(RsRsImmOperation):
    """
    Store 32-bit, values from the low bits of register rs2 to memory.

    M[x[rs1] + sext(offset)] = x[rs2][31:0]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sw
    """

    name = "riscv.sw"


## Assembler pseudo-instructions
## https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md


@irdl_op_definition
class LiOp(RdImmOperation):
    """
    Loads an immediate into rd.

    This is an assembler pseudo-instruction.

    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#load-immediate
    """

    name = "riscv.li"


RISCV = Dialect(
    [
        AddiOp,
        SltiOp,
        SltiuOp,
        AndiOp,
        OriOp,
        XoriOp,
        SlliOp,
        SrliOp,
        SraiOp,
        LuiOp,
        AuipcOp,
        AddOp,
        SltOp,
        SltuOp,
        AndOp,
        OrOp,
        XorOp,
        SllOp,
        SrlOp,
        SubOp,
        SraOp,
        NopOp,
        BeqOp,
        BneOp,
        BltOp,
        BgeOp,
        BltuOp,
        BgeuOp,
        LbOp,
        LbuOp,
        LhOp,
        LhuOp,
        LwOp,
        SbOp,
        ShOp,
        SwOp,
        LiOp,
    ],
    [RegisterType],
)
