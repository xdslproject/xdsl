"""
RISC-V 64-bit (RV64) dialect operations and types.

This module defines the RV64-specific variant of RISC-V operations,
using 6-bit immediates for 64-bit architectures.
"""

from __future__ import annotations

from xdsl.dialects.builtin import (
    I64,
    IntegerAttr,
    StringAttr,
    i64,
)
from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.dialects.builtin import I64, IntegerAttr, StringAttr, i64
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    LabelAttr,
    Registers,
    parse_immediate_value,
)
from xdsl.dialects.riscv.abstract_ops import (
    GetAnyRegisterOperation,
    ImmShiftOpHasCanonicalizationPatternsTrait,
    LiOperation,
    RdRsImmShiftOperation,
)
from xdsl.dialects.riscv.attrs import UI6, ui6
    LiOperation,
    RdRsImmIntegerOperation,
    RsRsImmIntegerOperation,
    assembly_arg_str,
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
    traits_def,
)
from xdsl.parser import Parser
from xdsl.traits import (
    AlwaysSpeculatable,
)


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


class RdRsImmShiftOperationRV64(RdRsImmShiftOperation[UI6, I64]):
    """Base class for RISC-V 64-bit shift immediate operations with rd, rs1 and imm6."""

    traits = lazy_traits_def(
        lambda: (ImmShiftOpRV64HasCanonicalizationPatternsTrait(),)
    )

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[UI6],
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, ui6)

        super().__init__(
            rs1=rs1,
            immediate=immediate,
            rd=rd,
            comment=comment,
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.immediate


class ImmShiftOpRV64HasCanonicalizationPatternsTrait(
    ImmShiftOpHasCanonicalizationPatternsTrait[I64],
    li_op_type=LiOp,
    shift_op_type=RdRsImmShiftOperationRV64,
):
    """Trait for RISC-V 64-bit shift immediate operations with canonicalization patterns."""


@irdl_op_definition
class SlliOp(RdRsImmShiftOperationRV64):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the 6-bit immediate.

    x[rd] = x[rs1] << shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli).
    """

    name = "rv64.slli"

    def py_operation(self, rs1: IntegerAttr[I64]) -> IntegerAttr[I64]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(rs1.value.data << self.immediate.value.data, i64)


@irdl_op_definition
class SrliOp(RdRsImmShiftOperationRV64):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the 6-bit immediate.

    x[rd] = x[rs1] >>u shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli).
    """

    name = "rv64.srli"

    def py_operation(self, rs1: IntegerAttr[I64]) -> IntegerAttr[I64]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(
            (rs1.value.data % 0x10000000000000000) >> self.immediate.value.data, i64
        )


@irdl_op_definition
class SraiOp(RdRsImmShiftOperationRV64):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the 6-bit immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "rv64.srai"

    def py_operation(self, rs1: IntegerAttr[I64]) -> IntegerAttr[I64]:
        assert isinstance(self.immediate, IntegerAttr)
        return IntegerAttr(rs1.value.data >> self.immediate.value.data, i64)


@irdl_op_definition
class SlliwOp(RdRsImmShiftOperationRV64):
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

    traits = traits_def(AlwaysSpeculatable())


@irdl_op_definition
class SrliwOp(RdRsImmShiftOperationRV64):
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

    traits = traits_def(AlwaysSpeculatable())


@irdl_op_definition
class LdOp(RdRsImmIntegerOperation):
    """
    Loads a 64-bit value from memory into register rd for RV64I.
    ```C
    x[rd] = M[x[rs1] + sext(offset)][63:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/#_ld).
    """

    name = "rv64.ld"

    traits = traits_def(MemoryReadEffect())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = assembly_arg_str(self.rd)
        imm = assembly_arg_str(self.immediate)
        base = assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({base})", self.comment
        )


@irdl_op_definition
class SdOp(RsRsImmIntegerOperation):
    """
    Store 64-bit, values from register rs2 to memory.
    ```C
    M[x[rs1] + sext(offset)] = x[rs2][63:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/#_sd).
    """

    name = "rv64.sd"

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = assembly_arg_str(self.rs2)
        imm = assembly_arg_str(self.immediate)
        base = assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({base})", self.comment
        )


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
        LdOp,
        SdOp,
        GetRegisterOp,
    ],
    [],
)
