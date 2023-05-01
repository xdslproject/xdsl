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
)

from xdsl.parser import BaseParser
from xdsl.printer import Printer


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


class RdRsRsOp(IRDLOperation, ABC):
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


@irdl_op_definition
class AddOp(RdRsRsOp):
    """
    Adds the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + x[rs2]

    https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#add
    """

    name = "riscv.add"


RISCV = Dialect(
    [
        AddOp,
    ],
    [RegisterType],
)
