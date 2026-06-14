"""
RISC-V 32-bit (RV32) dialect operations and types.

This module defines the RV32-specific variant of RISC-V operations,
using 5-bit immediates for 32-bit architectures.
"""

from __future__ import annotations

from xdsl.dialects.builtin import I32, IntegerAttr, StringAttr, i32
from xdsl.dialects.riscv import (
    UI5,
    AssemblyInstructionArg,
    IntRegisterType,
    LabelAttr,
    Registers,
    parse_immediate_value,
    print_immediate_value,
    ui5,
)
from xdsl.dialects.riscv.abstract_ops import GetAnyRegisterOperation, LiOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    Pure,
)
)
from xdsl.parser import Parser


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


class SlliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv32 import (
            ShiftLeftbyZero,
            ShiftLeftImmediate,
        )

        return (ShiftLeftImmediate(), ShiftLeftbyZero())


@irdl_op_definition
class SlliOp(RdRsImmShiftOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] << shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli).
    """

    name = "rv32.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())
    
    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        return IntegerAttr(rs1.value.data << self.immediate.value.data, i32)


class SrliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv32 import (
            ShiftRightbyZero,
            ShiftRightImmediate,
        )

        return (ShiftRightbyZero(), ShiftRightImmediate())


@irdl_op_definition
class SrliOp(RdRsImmShiftOperation):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>u shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli).
    """

    name = "rv32.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        return IntegerAttr(
            (rs1.value.data % 0x100000000) >> self.immediate.value.data, i32
        )

@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv32.srai"
    
    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        return IntegerAttr(rs1.value.data >> self.immediate.value.data, i32)


@irdl_op_definition
class LiOp(LiOperation[I32]):
    """
    Loads a 32-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv32.li"

    def __init__(
        self,
        immediate: int | IntegerAttr[I32] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i32)
        super().__init__(immediate, rd=rd, comment=comment)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i32)
        return attributes


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv32.get_register"


RV32 = Dialect(
    "rv32",
    [
        SlliOp,
        SrliOp,
        SraiOp,
        LiOp,
        LiOp,
        GetRegisterOp,
    ],
    [],
)
