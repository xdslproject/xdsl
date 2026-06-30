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
    ImmShiftOpHasCanonicalizationPatternsTrait,
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
    lazy_traits_def,
)
from xdsl.parser import Parser


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


class RV32RdRsImmShiftOperation(RdRsImmShiftOperation[UI5, I32]):
    """Base class for RISC-V 32-bit shift immediate operations with rd, rs1 and imm5."""

    traits = lazy_traits_def(
        lambda: (ImmShiftOpRV32HasCanonicalizationPatternsTrait(),)
    )

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


class ImmShiftOpRV32HasCanonicalizationPatternsTrait(
    ImmShiftOpHasCanonicalizationPatternsTrait[I32],
    li_op_type=LiOp,
    shift_op_type=RV32RdRsImmShiftOperation,
):
    """Trait for RISC-V 32-bit shift immediate operations with canonicalization patterns."""


@irdl_op_definition
class SlliOp(RV32RdRsImmShiftOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] << shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli).
    """

    name = "rv32.slli"

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(rs1.value.data << self.immediate.value.data, i32)


@irdl_op_definition
class SrliOp(RV32RdRsImmShiftOperation):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>u shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli).
    """

    name = "rv32.srli"

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(
            (rs1.value.data % 0x100000000) >> self.immediate.value.data, i32
        )


@irdl_op_definition
class BclriOp(RV32RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt.
    The index is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.

    See external [documentation](https://docs.riscv.org/reference/isa/v20260120/unpriv/b-st-ext.html#insns-bclri).
    """

    name = "rv32.bclri"

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(rs1.value.data & (~(1 << self.immediate.value.data)), i32)


@irdl_op_definition
class SraiOp(RV32RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv32.srai"

    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(rs1.value.data >> self.immediate.value.data, i32)


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "rv32.get_register"


RV32 = Dialect(
    "rv32",
    [
        SlliOp,
        SrliOp,
        SraiOp,
        BclriOp,
        LiOp,
        GetRegisterOp,
    ],
    [],
)
