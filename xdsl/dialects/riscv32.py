from __future__ import annotations

from abc import ABC
from collections.abc import Set as AbstractSet

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    attr_def,
    base,
    operand_def,
    result_def,
    traits_def,
)

from xdsl.irdl.operations import irdl_op_definition
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    Pure,
)

from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    LabelAttr,
    RISCVCustomFormatOperation,
    RISCVInstruction,
    IntRegisterType,
    Registers,
    SlliOpHasCanonicalizationPatternsTrait,
    SrliOpHasCanonicalizationPatternsTrait,
    UImm5Attr,
    parse_immediate_value,
    print_immediate_value,
    ui5,
)


class RdRsImmShiftOperationRV32(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """Base class for RISC-V 32-bit shift immediate operations with rd, rs1 and imm5."""

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(base(UImm5Attr) | base(LabelAttr))

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | UImm5Attr | str | LabelAttr,
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
class SlliRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SrliRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.srai"


@irdl_op_definition
class SlliwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.srliw"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.roriw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.slli.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.bexti"


@irdl_op_definition
class BinvIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.bseti"

    traits = traits_def(Pure())


RISCV32 = Dialect(
    "riscv32",
    [
        SlliRV32Op,
        SrliRV32Op,
        SraiRV32Op,
        # Bit Manipulation Operations
        BclrIRV32Op,
        BextIRV32Op,
        BinvIRV32Op,
        BsetIRV32Op,
    ],
    [],
)
