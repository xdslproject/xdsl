from __future__ import annotations


from dataclasses import dataclass, field
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
    A riscv register.
    """

    index: None = field(default=None)
    """
    The register index. Currently always None.
    """


@irdl_attr_definition
class RegisterType(Data[Register], TypeAttribute):
    name = "riscv.reg"

    @classmethod
    def new(cls, params: Register | None = None):
        # Create the new attribute object, without calling its __init__.
        # We do this to allow users to redefine their own __init__.
        attr = cls.__new__(cls)

        if params is None:
            register = Register(None)
        elif isinstance(params, Register):
            register = params

        # Call the __init__ of Data, which will set the parameters field.
        Data[Register].__init__(attr, register)
        return attr

    @staticmethod
    def parse_parameter(parser: BaseParser) -> Register:
        return Register(None)

    def print_parameter(self, printer: Printer) -> None:
        pass


class Riscv1Rd2RsOperation(IRDLOperation):
    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: RegisterType | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())

        super().__init__(
            operands=[rs1, rs2],
            result_types=[rd],
        )


@irdl_op_definition
class AddOp(Riscv1Rd2RsOperation):
    name = "riscv.add"


RISCV = Dialect(
    [
        AddOp,
    ],
    [RegisterType],
)
