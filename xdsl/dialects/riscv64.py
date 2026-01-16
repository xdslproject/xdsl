from __future__ import annotations

from abc import ABC
from collections.abc import Set as AbstractSet
from typing import Annotated

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    Signedness,
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
    parse_immediate_value,
    print_immediate_value,
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

ui6 = IntegerType(6, Signedness.UNSIGNED)
UImm6Attr = IntegerAttr[Annotated[IntegerType, ui6]]


class RdRsImmShiftOperationRV64(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """Base class for RISC-V 64-bit shift immediate operations with rd, rs1 and imm6."""

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(base(UImm6Attr) | base(LabelAttr))

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | UImm6Attr | str | LabelAttr,
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


@irdl_op_definition
class SlliOp(RdRsImmShiftOperationRV64):
    name = "riscv64.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SrliOp(RdRsImmShiftOperationRV64):
    name = "riscv64.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiOp(RdRsImmShiftOperationRV64):
    name = "riscv64.srai"


@irdl_op_definition
class SlliwOp(RdRsImmShiftOperationRV64):
    name = "riscv64.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwOp(RdRsImmShiftOperationRV64):
    name = "riscv64.srliw"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriOp(RdRsImmShiftOperationRV64):
    name = "riscv64.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriwOp(RdRsImmShiftOperationRV64):
    name = "riscv64.roriw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwOp(RdRsImmShiftOperationRV64):
    name = "riscv64.slli.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperationRV64):
    name = "riscv64.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperationRV64):
    name = "riscv64.bexti"


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperationRV64):
    name = "riscv64.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperationRV64):
    name = "riscv64.bseti"

    traits = traits_def(Pure())


RISCV64 = Dialect(
    "riscv64",
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
