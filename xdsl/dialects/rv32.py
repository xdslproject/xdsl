"""
RISC-V 32-bit (RV32) dialect operations and types.

This module defines the RV32-specific variant of RISC-V operations,
using 5-bit immediates for 32-bit architectures.
"""

from __future__ import annotations

from xdsl.dialects.builtin import I32, IntegerAttr, StringAttr, i32
from xdsl.dialects.riscv import (
    UI5,
    IntRegisterType,
    LabelAttr,
    Registers,
    parse_immediate_value,
    ui5,
)
from xdsl.dialects.riscv.abstract_ops import (
    GetAnyRegisterOperation,
    LiOperation,
    RdRsImmShiftOperation,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    irdl_op_definition,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
)


class RdRsImmShiftOperationRV32(RdRsImmShiftOperation[UI5, I32]):
    """Base class for RISC-V 32-bit shift immediate operations with rd, rs1 and imm5."""

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[UI5],
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, ui5)

        super().__init__(
            rs1=rs1,
            immediate=immediate,
            rd=rd,
            comment=comment,
        )


class SlliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ShiftbyZero,
            ShiftConstantFolding,
        )

        return (ShiftbyZero(), ShiftConstantFolding())


@irdl_op_definition
class SlliOp(RdRsImmShiftOperationRV32):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] << shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli).
    """

    name = "rv32.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(rs1.value.data << self.immediate.value.data, i32)


class SrliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ShiftbyZero,
            ShiftConstantFolding,
        )

        return (ShiftbyZero(), ShiftConstantFolding())


@irdl_op_definition
class SrliOp(RdRsImmShiftOperationRV32):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>u shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli).
    """

    name = "rv32.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(
            (rs1.value.data % 0x100000000) >> self.immediate.value.data, i32
        )

class SraiOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ShiftbyZero,
            ShiftConstantFolding,
        )

        return (ShiftbyZero(), ShiftConstantFolding())


@irdl_op_definition
class SraiOp(RdRsImmShiftOperationRV32):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv32.srai"

    traits = traits_def(SraiOpHasCanonicalizationPatternsTrait())

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
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
        GetRegisterOp,
    ],
    [],
)
