"""
RISC-V 64-bit (RV64) dialect operations and types.

This module defines the RV64-specific variant of RISC-V operations,
using 6-bit immediates for 64-bit architectures.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from typing import Literal, TypeAlias

from xdsl.dialects.builtin import I64, IntegerAttr, StringAttr, i64
from xdsl.dialects.riscv import (
    IntRegisterType,
    LabelAttr,
    Registers,
    parse_immediate_value,
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

UI6: TypeAlias = IntegerType[Literal[6], Literal[Signedness.UNSIGNED]]
ui6: UI6 = IntegerType(6, Signedness.UNSIGNED)


class RdRsImmShiftOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """Base class for RISC-V 64-bit shift immediate operations with rd, rs1 and imm6."""

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[UI6] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[UI6] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, ui6)
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
        attributes["immediate"] = parse_immediate_value(parser, ui6)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class SlliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            ShiftLeftbyZero,
            ShiftLeftImmediate,
        )

        return (ShiftLeftImmediate(), ShiftLeftbyZero())


@irdl_op_definition
class SlliOp(RdRsImmShiftOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the 6-bit immediate.

    x[rd] = x[rs1] << shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli).
    """

    name = "rv64.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())
    
    def __ini
    
    def py_operation(self, rs1: IntegerAttr[I64]) -> IntegerAttr[I64]:
        return IntegerAttr(rs1.value.data << self.immediate.value.data, i64)


class SrliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            ShiftRightbyZero,
            ShiftRightImmediate,
        )

        return (ShiftRightbyZero(), ShiftRightImmediate())


@irdl_op_definition
class SrliOp(RdRsImmShiftOperation):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the 6-bit immediate.

    x[rd] = x[rs1] >>u shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli).
    """

    name = "rv64.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())
    
    def py_operation(self, rs1: IntegerAttr[I64]) -> IntegerAttr[I64]:
        return IntegerAttr(
            (rs1.value.data % 0x100000000) >> self.immediate.value.data, i64
        )


@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the 6-bit immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv64.srai"
    
    def py_operation(self, rs1: IntegerAttr[I64]) -> IntegerAttr[I64]:
        return IntegerAttr(rs1.value.data >> self.immediate.value.data, i64)


@irdl_op_definition
class SlliwOp(RdRsImmShiftOperation):
    """
    Performs logical left shift on the lower 32 bits of the value in register rs1
    by the shift amount held in the immediate (RV64-only instruction).
    The result is sign-extended to 64 bits.
    ```
    x[rd] = sext((x[rs1] << shamt)[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64i.html#slliw).
    """

    name = "rv64.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwOp(RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the 32-bit of value in register rs1
    by the shift amount held in the lower 5 bits of the immediate. (RV64-only instruction).
    The result is sign-extended to 64 bits.
    ```
    x[rd] = sext((x[rs1] << shamt)[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64i.html#srliw).
    """

    name = "rv64.srliw"

    traits = traits_def(Pure())


@irdl_op_definition
class LiOp(LiOperation[I64]):
    """
    Loads a 64-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv64.li"

    def __init__(
        self,
        immediate: int | IntegerAttr[I64] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i64)
        super().__init__(immediate, rd=rd, comment=comment)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i64)
        return attributes


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv64.get_register"


RV64 = Dialect(
    "rv64",
    [
        SlliOp,
        SrliOp,
        SraiOp,
        SlliwOp,
        SrliwOp,
        LiOp,
        LiOp,
        GetRegisterOp,
    ],
    [],
)
