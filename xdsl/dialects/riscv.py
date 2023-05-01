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


# Arithmetic


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
class SubOp(RdRsRsOperation):
    """
    Subtracts the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] - x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub
    """

    name = "riscv.sub"


@irdl_op_definition
class LiOp(RdImmOperation):
    """
    Loads an immediate into rd.

    This is an assembler pseudo-instruction.

    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#load-immediate
    """

    name = "riscv.li"


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


# Logical


@irdl_op_definition
class XorOp(RdRsRsOperation):
    """
    Performs bitwise XOR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] ^ x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xor
    """

    name = "riscv.xor"


RISCV = Dialect(
    [
        AddOp,
        SubOp,
        LiOp,
        XorOp,
        AddiOp,
        SltiOp,
    ],
    [RegisterType],
)
