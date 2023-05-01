from __future__ import annotations


from dataclasses import dataclass, field
from typing import Annotated

from xdsl.ir import (
    Dialect,
    Operation,
    SSAValue,
    Data,
    OpResult,
)

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    irdl_attr_definition,
    OptOpAttr,
    Operand,
)
from xdsl.dialects.builtin import StringAttr

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
class RegisterType(Data[Register]):
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
        assert False

    def print_parameter(self, printer: Printer) -> None:
        assert False


class Riscv1Rd2RsOperation(IRDLOperation):
    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    comment: OptOpAttr[StringAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: RegisterType | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "comment": comment,
            },
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
