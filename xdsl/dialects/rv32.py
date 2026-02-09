"""
RISC-V 32-bit (RV32) dialect operations and types.

This module defines the RV32-specific variant of RISC-V operations,
using 5-bit immediates for 32-bit architectures.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from xdsl.dialects.builtin import I32, IntegerAttr, StringAttr, i32
from xdsl.dialects.riscv import (
    UI5,
    AssemblyInstructionArg,
    IntRegisterType,
    LabelAttr,
    Registers,
    RISCVCustomFormatOperation,
    RISCVInstruction,
    parse_immediate_value,
    print_immediate_value,
    ui5,
)
from xdsl.dialects.riscv.ops import (
    LiOpHasCanonicalizationPatternTrait,
    SlliOpHasCanonicalizationPatternsTrait,
    SrliOpHasCanonicalizationPatternsTrait,
)
from xdsl.interfaces import ConstantLikeInterface
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    Pure,
)


class RdRsImmShiftOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """Base class for RISC-V 32-bit shift immediate operations with rd, rs1 and imm5."""

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[UI5] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[UI5] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, ui5)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)

        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            result_types=[rd],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, ui5)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class SlliOp(RdRsImmShiftOperation):
    name = "rv32.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SrliOp(RdRsImmShiftOperation):
    name = "rv32.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    name = "rv32.srai"


@irdl_op_definition
class RoriOp(RdRsImmShiftOperation):
    name = "rv32.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperation):
    name = "rv32.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperation):
    name = "rv32.bexti"

    traits = traits_def(Pure())


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperation):
    name = "rv32.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperation):
    name = "rv32.bseti"

    traits = traits_def(Pure())


@irdl_op_definition
class LiOp(RISCVCustomFormatOperation, RISCVInstruction, ConstantLikeInterface, ABC):
    """
    Loads a 32-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv32.li"

    rd = result_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[I32] | LabelAttr)

    traits = traits_def(Pure(), LiOpHasCanonicalizationPatternTrait())

    def __init__(
        self,
        immediate: int | IntegerAttr[I32] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i32)
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

    def get_constant_value(self):
        return self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i32)
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


RV32 = Dialect(
    "rv32",
    [
        SlliOp,
        SrliOp,
        SraiOp,
        RoriOp,
        # Bit Manipulation Operations
        BclrIOp,
        BextIOp,
        BinvIOp,
        BsetIOp,
        LiOp,
    ],
    [],
)
