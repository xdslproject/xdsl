from __future__ import annotations

from abc import ABC
from collections.abc import Set as AbstractSet

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
)
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    LabelAttr,
    Registers,
    RISCVCustomFormatOperation,
    RISCVInstruction,
    SlliOpHasCanonicalizationPatternsTrait,
    SrliOpHasCanonicalizationPatternsTrait,
    UImm5Attr,
    parse_immediate_value,
    print_immediate_value,
    ui5,
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
class SlliOp(RdRsImmShiftOperationRV32):
    name = "rv32.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SrliOp(RdRsImmShiftOperationRV32):
    name = "rv32.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiOp(RdRsImmShiftOperationRV32):
    name = "rv32.srai"


@irdl_op_definition
class SlliwOp(RdRsImmShiftOperationRV32):
    name = "rv32.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwOp(RdRsImmShiftOperationRV32):
    name = "rv32.srliw"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriOp(RdRsImmShiftOperationRV32):
    name = "rv32.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriwOp(RdRsImmShiftOperationRV32):
    name = "rv32.roriw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwOp(RdRsImmShiftOperationRV32):
    name = "rv32.slli.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperationRV32):
    name = "rv32.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperationRV32):
    name = "rv32.bexti"


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperationRV32):
    name = "rv32.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperationRV32):
    name = "rv32.bseti"

    traits = traits_def(Pure())


RV32 = Dialect(
    "rv32",
    [
        SlliOp,
        SrliOp,
        SraiOp,
        # Bit Manipulation Operations
        BclrIOp,
        BextIOp,
        BinvIOp,
        BsetIOp,
    ],
    [],
)
