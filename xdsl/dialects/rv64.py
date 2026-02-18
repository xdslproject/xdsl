"""
RISC-V 64-bit (RV64) dialect operations and types.

This module defines the RV64-specific variant of RISC-V operations,
using 6-bit immediates for 64-bit architectures.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from xdsl.dialects.builtin import I64, IntegerAttr, StringAttr, i64
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    LabelAttr,
    Registers,
    RISCVCustomFormatOperation,
    RISCVInstruction,
    parse_immediate_value,
    print_immediate_value,
)
from xdsl.dialects.riscv.abstract_ops import GetAnyRegisterOperation
from xdsl.dialects.riscv.ops import LiOpHasCanonicalizationPatternTrait
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    attr_def,
    irdl_op_definition,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    ConstantLike,
    Pure,
)


@irdl_op_definition
class LiOp(RISCVCustomFormatOperation, RISCVInstruction, HasFolderInterface, ABC):
    """
    Loads a 64-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv64.li"

    rd = result_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[I64] | LabelAttr)

    traits = traits_def(Pure(), LiOpHasCanonicalizationPatternTrait(), ConstantLike())

    def __init__(
        self,
        immediate: int | IntegerAttr[I64] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i64)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            result_types=[rd],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.immediate

    def fold(self) -> tuple[IntegerAttr[I64] | LabelAttr]:
        return (self.immediate,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i64)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        print_immediate_value(printer, self.immediate)
        return {"immediate", "fastmath"}

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        return (), (res_type,)

    def print_op_type(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.rd.type)


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv64.get_register"


RV64 = Dialect(
    "rv64",
    [
        LiOp,
        GetRegisterOp,
    ],
    [],
)
