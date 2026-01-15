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
    parse_immediate_value,
    print_immediate_value,
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
class SlliRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SrliRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.srai"


@irdl_op_definition
class SlliwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.srliw"

    traits = traits_def(Pure())
    

@irdl_op_definition
class RoriRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.rori"
    
    traits = traits_def(Pure())
    

@irdl_op_definition
class RoriwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.roriw"
    
    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.slli.uw"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BclrIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.bclri"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BextIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.bexti"


@irdl_op_definition
class BinvIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.binvi"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BsetIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.bseti"

    traits = traits_def(Pure())


RISCV64 = Dialect(
    "riscv64",
    [
        SlliRV64Op,
        SrliRV64Op,
        SraiRV64Op,
        # Bit Manipulation Operations
        BclrIRV64Op,
        BextIRV64Op,
        BinvIRV64Op,
        BsetIRV64Op,
    ],
    [],
)
