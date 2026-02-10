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
from xdsl.dialects.riscv.ops import LiOpHasCanonicalizationPatternTrait
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
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
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


@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv32.srai"


@irdl_op_definition
class RoriOp(RdRsImmShiftOperation):
    """
    Right rotation of rs1 by the amount in the least-significant
    log2(XLEN) bits of shamt. For RV32, the encodings corresponding to shamt[5]=1 are reserved.
    ```
    let shamt = if   xlen == 32
                    then shamt[4..0]
                    else shamt[5..0];
    let result = (X(rs1) >> shamt) | (X(rs2) << (xlen - shamt));
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-rori).
    """

    name = "rv32.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt.
    The index is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = X(rs1) & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bclri).
    """

    name = "rv32.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperation):
    """
    This instruction returns a single bit extracted from rs1 at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = (X(rs1) >> index) & 1;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bexti).
    """

    name = "rv32.bexti"


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt. The index
    is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    x[rd] = x[rs1] & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-binvi).
    """

    name = "rv32.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit set at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of rs2.
    ```
    let index = X(rs2) & (XLEN - 1);
    X(rd) = X(rs1) | (1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bset).
    """

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
        BclrIOp,
        BextIOp,
        BinvIOp,
        BsetIOp,
        LiOp,
    ],
    [],
)
