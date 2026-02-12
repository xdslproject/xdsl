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

from xdsl.dialects.builtin import (
    I64,
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    i64,
)
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


@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the 6-bit immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv64.srai"


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
class RoriOp(RdRsImmShiftOperation):
    """
    Right rotation of rs1 by the amount in the least-significant
    log2(XLEN) bits of shamt.
    ```
    let shamt = if   xlen == 32
                    then shamt[4..0]
                    else shamt[5..0];
    let result = (X(rs1) >> shamt) | (X(rs2) << (xlen - shamt));
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-rori).
    """

    name = "rv64.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriwOp(RdRsImmShiftOperation):
    """
    Right rotation on the least-significant word of rs1 by the amount in
    the least-significant log2(XLEN) bits of shamt. The resulting word value is sign-extended by
    copying bit 31 to all of the more-significant bits.
    ```
    let rs1 = EXTZ(X(rs1)[31..0];
    let result = (rs1 >> shamt[4..0]) | (X(rs1) << (32 - shamt[4..0]));
    X(rd) = EXTS(result[31..0]);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-roriw).
    """

    name = "rv64.roriw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwOp(RdRsImmShiftOperation):
    """
    This instruction takes the least-significant word of rs1, zero-extends it,
    and shifts it left by the immediate.
    ```
    x[rd] = (EXTZ(x[rs][31..0]) << shamt);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-slli_uw).
    """

    name = "rv64.slli.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt.
    The index is read from the lower log2(XLEN) bits of shamt.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = X(rs1) & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bclri).
    """

    name = "rv64.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperation):
    """
    This instruction returns a single bit extracted from rs1 at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of shamt.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = (X(rs1) >> index) & 1;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bexti).
    """

    name = "rv64.bexti"


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt. The index
    is read from the lower log2(XLEN) bits of shamt.
    ```
    let index = shamt & (XLEN - 1);
    x[rd] = x[rs1] & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-binvi).
    """

    name = "rv64.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit set at the index specified in shamt. The index is read
    from the lower log2(XLEN) bits of shamt.
    ```
    let index = shamt & (XLEN - 1);
    x[rd] = x[rs1] | (1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bseti).
    """

    name = "rv64.bseti"

    traits = traits_def(Pure())


@irdl_op_definition
class LiOp(RISCVCustomFormatOperation, RISCVInstruction, ConstantLikeInterface, ABC):
    """
    Loads a 64-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/main/src/asm-manual.adoc).
    """

    name = "rv64.li"

    rd = result_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[I64] | LabelAttr)

    traits = traits_def(Pure(), LiOpHasCanonicalizationPatternTrait())

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

    def get_constant_value(self):
        return self.immediate

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


RV64 = Dialect(
    "rv64",
    [
        SlliOp,
        SrliOp,
        SraiOp,
        SlliwOp,
        SrliwOp,
        RoriOp,
        RoriwOp,
        SlliUwOp,
        BclrIOp,
        BextIOp,
        BinvIOp,
        BsetIOp,
        LiOp,
    ],
    [],
)
