"""
RISC-V 64-bit (RV64) dialect operations and types.

This module defines the RV64-specific variant of RISC-V operations,
using 6-bit shift immediates for 64-bit architectures.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Set as AbstractSet
from typing import Literal, TypeAlias

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
)
from xdsl.dialects.riscv import (
    SI12,
    AssemblyInstructionArg,
    IntRegisterType,
    LabelAttr,
    Registers,
    RISCVCustomFormatOperation,
    RISCVInstruction,
    parse_immediate_value,
    print_immediate_value,
    si12,
)
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


class RdRsImmIntegerOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V 64-bit operations that have one destination register, one source
    register and one immediate operand.

    This is called I-Type in the RISC-V specification.
    """

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[SI12] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[SI12] | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, si12)
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
        attributes["immediate"] = parse_immediate_value(parser, si12)
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
    name = "rv64.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    name = "rv64.srai"


@irdl_op_definition
class SlliwOp(RdRsImmShiftOperation):
    name = "rv64.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwOp(RdRsImmShiftOperation):
    name = "rv64.srliw"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriOp(RdRsImmShiftOperation):
    name = "rv64.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriwOp(RdRsImmShiftOperation):
    name = "rv64.roriw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwOp(RdRsImmShiftOperation):
    name = "rv64.slli.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperation):
    name = "rv64.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperation):
    name = "rv64.bexti"


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperation):
    name = "rv64.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperation):
    name = "rv64.bseti"

    traits = traits_def(Pure())


class AddiOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            AddImmediateConstant,
            AddImmediateZero,
        )

        return (
            AddImmediateZero(),
            AddImmediateConstant(),
        )


@irdl_op_definition
class AddiOp(RdRsImmIntegerOperation):
    """
    Adds the sign-extended 12-bit immediate to register rs1.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#addi).
    """

    name = "rv64.addi"

    traits = traits_def(Pure(), AddiOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SltiOp(RdRsImmIntegerOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the sign-extended
    immediate when both are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slti).
    """

    name = "rv64.slti"


@irdl_op_definition
class SltiuOp(RdRsImmIntegerOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the immediate when
    both are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltiu).
    """

    name = "rv64.sltiu"


class AndiOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            AndiImmediate,
        )

        return (AndiImmediate(),)


@irdl_op_definition
class AndiOp(RdRsImmIntegerOperation):
    """
    Performs bitwise AND on register rs1 and the sign-extended 12-bit
    immediate and place the result in rd.

    x[rd] = x[rs1] & sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#andi).
    """

    name = "rv64.andi"
    traits = traits_def(AndiOpHasCanonicalizationPatternsTrait())


class OriOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            OriImmediate,
        )

        return (OriImmediate(),)


@irdl_op_definition
class OriOp(RdRsImmIntegerOperation):
    """
    Performs bitwise OR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] | sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#ori).
    """

    name = "rv64.ori"
    traits = traits_def(OriOpHasCanonicalizationPatternsTrait())


class XoriOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            XoriImmediate,
        )

        return (XoriImmediate(),)


@irdl_op_definition
class XoriOp(RdRsImmIntegerOperation):
    """
    Performs bitwise XOR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] ^ sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xori).
    """

    name = "rv64.xori"
    traits = traits_def(XoriOpHasCanonicalizationPatternsTrait())


# Addiw and Sraiw are RV64I-specific Instructions


@irdl_op_definition
class AddiwOp(RdRsImmIntegerOperation):
    """
    Adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd.
    Overflows are ignored and the result is the low 32 bits of the result sign-extended to 64 bits.
    ```
    x[rd] = sext((x[rs1] + sext(immediate))[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64i.html#addiw).
    """

    name = "rv64.addiw"

    traits = traits_def(Pure())


@irdl_op_definition
class SraiwOp(RdRsImmIntegerOperation):
    """
    Performs arithmetic right shift on the 32-bit of value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.
    ```
    x[rd] = sext(x[rs1][31:0] >>s shamt)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sraiw).
    """

    name = "rv64.sraiw"

    traits = traits_def(Pure())


@irdl_op_definition
class LbOp(RdRsImmIntegerOperation):
    """
    Loads a 8-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = sext(M[x[rs1] + sext(offset)][7:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lb).
    """

    name = "rv64.lb"


@irdl_op_definition
class LbuOp(RdRsImmIntegerOperation):
    """
    Loads a 8-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = M[x[rs1] + sext(offset)][7:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lbu).
    """

    name = "rv64.lbu"


@irdl_op_definition
class LhOp(RdRsImmIntegerOperation):
    """
    Loads a 16-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = sext(M[x[rs1] + sext(offset)][15:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lh).
    """

    name = "rv64.lh"


@irdl_op_definition
class LhuOp(RdRsImmIntegerOperation):
    """
    Loads a 16-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = M[x[rs1] + sext(offset)][15:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lhu).
    """

    name = "rv64.lhu"


class LwOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.rv64 import (
            LoadWordWithKnownOffset,
        )

        return (LoadWordWithKnownOffset(),)


@irdl_op_definition
class LwOp(RdRsImmIntegerOperation):
    """
    Loads a 32-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = sext(M[x[rs1] + sext(offset)][31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lw).
    """

    name = "rv64.lw"

    traits = traits_def(LwOpHasCanonicalizationPatternTrait())


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
        # Bit Manipulation Operations
        BclrIOp,
        BextIOp,
        BinvIOp,
        BsetIOp,
        # RdRsImmIntegerOperation Operations
        AddiOp,
        SltiOp,
        SltiuOp,
        AndiOp,
        OriOp,
        XoriOp,
        AddiwOp,
        SraiwOp,
        LbOp,
        LbuOp,
        LhOp,
        LhuOp,
        LwOp,
    ],
    [],
)
