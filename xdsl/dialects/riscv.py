from __future__ import annotations
from abc import ABC

from dataclasses import dataclass, field
from typing import Annotated, Iterable

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
    RISC-V registers have an index between 0 and 31. A value of `None` means that the
    register has not yet been allocated.
    """

    index: int | None = field(default=None)
    """The register index. Can be between 0 and 31, or None."""

    ABI_NAMES = [
        "zero",
        "ra",
        "sp",
        "gp",
        "tp",
        "t0",
        "t1",
        "t2",
        "fp",
        "s0",
        "s1",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "t3",
        "t4",
        "t5",
        "t6",
    ]

    ABI_INDEX_BY_NAME = {name: index for index, name in enumerate(ABI_NAMES)}

    @staticmethod
    def from_name(name: str) -> Register:
        try:
            index = Register.ABI_INDEX_BY_NAME[name]
            return Register(index)
        except KeyError:
            raise ValueError(f"Unknown register name: {name}")

    @property
    def abi_name(self) -> str | None:
        if self.index is None:
            return None
        return Register.ABI_NAMES[self.index]


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
        name = self.data.abi_name
        if name is None:
            name = "x?"
        printer.print_string(name)


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "riscv.label"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> str:
        assert False

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(self.data)


class _RegisterAllocation:
    register_by_value: dict[SSAValue, str] = {}
    idx = 0

    def __init__(self) -> None:
        self.register_by_value = dict()
        self.idx = 0

    def register_for_value(self, reg: SSAValue):
        if reg in self.register_by_value:
            name = self.register_by_value[reg]
        else:
            typ = reg.typ
            if not isinstance(typ, RegisterType):
                raise ValueError(f"Expected RegisterType, got {typ}")
            name = typ.data.abi_name
            if name is None:
                name = f"%{self.idx}"
                self.idx += 1
            self.register_by_value[reg] = name
        return name


class _RISCVOp(Operation, ABC):
    def assembly_instruction(
        self, register_allocation: _RegisterAllocation
    ) -> str | None:
        assert self.name.startswith("riscv.")
        comment: StringAttr | None
        match self:
            case Riscv1Rd2RsOperation():
                instruction_name = self.name[6:]
                comment = self.comment
            case _:
                assert False, f"Unknown operation type {type(self)}"

        components: list[str] = []

        reg = register_allocation.register_for_value

        for result in self.results:
            components.append(reg(result))

        for operand in self.operands:
            components.append(reg(operand))

        code = f"    {instruction_name} {', '.join(components)}"
        return _RISCVOp.append_comment(code, comment)

    @staticmethod
    def append_comment(line: str, comment: StringAttr | None) -> str:
        if comment is None:
            return line

        padding = " " * max(0, 48 - len(line))

        return f"{line}{padding} # {comment.data}"


def riscv_code(ops: Iterable[Operation]) -> str:
    code = ""

    reg = _RegisterAllocation()

    for op in ops:
        if not isinstance(op, _RISCVOp):
            raise ValueError("All ops in module must inherit from RISCVOp")
        instruction = op.assembly_instruction(reg)
        if instruction is None:
            continue

        code += f"{instruction}\n"

    return code


class Riscv1Rd2RsOperation(IRDLOperation, _RISCVOp):
    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    comment: OptOpAttr[StringAttr]

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: RegisterType | str | None = None,
        comment: str | StringAttr | None = None,
    ):
        if rd is None:
            rd = RegisterType(Register())
        elif isinstance(rd, str):
            rd = RegisterType(Register.from_name(rd))
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
