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
    AddiOp,
    AddiwOp,
    AndiOp,
    IntRegisterType,
    RdRsIntegerOperation,
    RdRsRsIntegerOperation,
    AuipcOp,
    BeqOp,
    BgeOp,
    BgeuOp,
    BltOp,
    BltuOp,
    BneOp,
    CsrrcOp,
    CsrrciOp,
    CsrrsOp,
    CsrrsiOp,
    CsrrwOp,
    CsrrwiOp,
    CustomAssemblyInstructionOp,
    LiOp,
    DirectiveOp,
    EbreakOp,
    EcallOp,
    JalOp,
    JalrOp,
    JOp,
    LabelOp,
    LbOp,
    LbuOp,
    LhOp,
    LhuOp,
    LuiOp,
    LwOp,
    NopOp,
    OriOp,
    Registers,
    ReturnOp,
    SbOp,
    ShOp,
    SlliOpHasCanonicalizationPatternsTrait,
    SltiOp,
    SltiuOp,
    SraiwOp,
    SrliOpHasCanonicalizationPatternsTrait,
    SwOp,
    WfiOp,
    XoriOp,
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
class AddiRV64Op(AddiOp):
    name = "riscv64.addi"


@irdl_op_definition
class SltiRV64Op(SltiOp):
    name = "riscv64.slti"


@irdl_op_definition
class SltiuRV64Op(SltiuOp):
    name = "riscv64.sltiu"


@irdl_op_definition
class AndiRV64Op(AndiOp):
    name = "riscv64.andi"


@irdl_op_definition
class OriRV64Op(OriOp):
    name = "riscv64.ori"


@irdl_op_definition
class XoriRV64Op(XoriOp):
    name = "riscv64.xori"


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
class AddiwRV64Op(AddiwOp):
    name = "riscv64.addiw"

@irdl_op_definition
class SlliwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.slliw"

    traits = traits_def(Pure())

@irdl_op_definition
class SrliwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.srliw"

    traits = traits_def(Pure())

@irdl_op_definition
class SraiwRV64Op(SraiwOp):
    name = "riscv64.sraiw"


@irdl_op_definition
class AddwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.addw"
    traits = traits_def(Pure())


@irdl_op_definition
class SubwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.subw"
    traits = traits_def(Pure())


@irdl_op_definition
class SllwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sllw"
    traits = traits_def(Pure())


@irdl_op_definition
class SrlwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.srlw"
    traits = traits_def(Pure())


@irdl_op_definition
class SrawRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sraw"
    traits = traits_def(Pure())


@irdl_op_definition
class LuiRV64Op(LuiOp):
    name = "riscv64.lui"


@irdl_op_definition
class AuipcRV64Op(AuipcOp):
    name = "riscv64.auipc"


@irdl_op_definition
class SeqzRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.seqz"


@irdl_op_definition
class SnezRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.snez"


@irdl_op_definition
class ZextBRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.zext.b"
    traits = traits_def(Pure())


@irdl_op_definition
class ZextWRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.zext.w"
    traits = traits_def(Pure())


@irdl_op_definition
class SextWRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.sext.w"
    traits = traits_def(Pure())


@irdl_op_definition
class AddRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.add"
    traits = traits_def(Pure())


@irdl_op_definition
class SltRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.slt"


@irdl_op_definition
class SltuRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sltu"


@irdl_op_definition
class AndRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.and"
    traits = traits_def(Pure())


@irdl_op_definition
class OrRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.or"
    traits = traits_def(Pure())


@irdl_op_definition
class XorRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.xor"
    traits = traits_def(Pure())


@irdl_op_definition
class SllRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sll"
    traits = traits_def(Pure())


@irdl_op_definition
class SrlRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.srl"
    traits = traits_def(Pure())


@irdl_op_definition
class SubRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sub"
    traits = traits_def(Pure())


@irdl_op_definition
class SraRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sra"
    traits = traits_def(Pure())


@irdl_op_definition
class NopRV64Op(NopOp):
    name = "riscv64.nop"


@irdl_op_definition
class JalRV64Op(JalOp):
    name = "riscv64.jal"


@irdl_op_definition
class JRV64Op(JOp):
    name = "riscv64.j"


@irdl_op_definition
class JalrRV64Op(JalrOp):
    name = "riscv64.jalr"


@irdl_op_definition
class ReturnRV64Op(ReturnOp):
    name = "riscv64.ret"


@irdl_op_definition
class BeqRV64Op(BeqOp):
    name = "riscv64.beq"


@irdl_op_definition
class BneRV64Op(BneOp):
    name = "riscv64.bne"


@irdl_op_definition
class BltRV64Op(BltOp):
    name = "riscv64.blt"


@irdl_op_definition
class BgeRV64Op(BgeOp):
    name = "riscv64.bge"


@irdl_op_definition
class BltuRV64Op(BltuOp):
    name = "riscv64.bltu"


@irdl_op_definition
class BgeuRV64Op(BgeuOp):
    name = "riscv64.bgeu"


@irdl_op_definition
class LbRV64Op(LbOp):
    name = "riscv64.lb"


@irdl_op_definition
class LbuRV64Op(LbuOp):
    name = "riscv64.lbu"


@irdl_op_definition
class LhRV64Op(LhOp):
    name = "riscv64.lh"


@irdl_op_definition
class LhuRV64Op(LhuOp):
    name = "riscv64.lhu"


@irdl_op_definition
class LwRV64Op(LwOp):
    name = "riscv64.lw"


@irdl_op_definition
class SbRV64Op(SbOp):
    name = "riscv64.sb"


@irdl_op_definition
class ShRV64Op(ShOp):
    name = "riscv64.sh"


@irdl_op_definition
class SwRV64Op(SwOp):
    name = "riscv64.sw"


@irdl_op_definition
class CsrrwRV64Op(CsrrwOp):
    name = "riscv64.csrrw"


@irdl_op_definition
class CsrrsRV64Op(CsrrsOp):
    name = "riscv64.csrrs"


@irdl_op_definition
class CsrrcRV64Op(CsrrcOp):
    name = "riscv64.csrrc"


@irdl_op_definition
class CsrrwiRV64Op(CsrrwiOp):
    name = "riscv64.csrrwi"


@irdl_op_definition
class CsrrsiRV64Op(CsrrsiOp):
    name = "riscv64.csrrsi"


@irdl_op_definition
class CsrrciRV64Op(CsrrciOp):
    name = "riscv64.csrrci"


@irdl_op_definition
class MulRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.mul"
    traits = traits_def(Pure())


@irdl_op_definition
class MulhRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.mulh"
    traits = traits_def(Pure())


@irdl_op_definition
class MulhsuRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.mulhsu"
    traits = traits_def(Pure())


@irdl_op_definition
class MulhuRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.mulhu"
    traits = traits_def(Pure())


@irdl_op_definition
class MulwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.mulw"
    traits = traits_def(Pure())


@irdl_op_definition
class DivRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.div"


@irdl_op_definition
class DivuRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.divu"


@irdl_op_definition
class DivuwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.divuw"


@irdl_op_definition
class DivwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.divw"


@irdl_op_definition
class RemRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.rem"


@irdl_op_definition
class RemuRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.remu"


@irdl_op_definition
class RemuwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.remuw"


@irdl_op_definition
class RemwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.remw"


@irdl_op_definition
class RolRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.rol"
    traits = traits_def(Pure())

@irdl_op_definition
class RorRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.ror"
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
class SextHRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.sext.h"
    traits = traits_def(Pure())


@irdl_op_definition
class ZextHRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.zext.h"
    traits = traits_def(Pure())


@irdl_op_definition
class SextBRV64Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv64.sext.b"
    traits = traits_def(Pure())


@irdl_op_definition
class RolwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.rolw"
    traits = traits_def(Pure())


@irdl_op_definition
class RorwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.rorw"
    traits = traits_def(Pure())


@irdl_op_definition
class AddUwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh1addRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sh1add"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh2addRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sh2add"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh3addRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sh3add"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh1addUwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sh1add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh2addUwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sh2add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh3addUwRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.sh3add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.slli.uw"
    
    traits = traits_def(Pure())

@irdl_op_definition
class CZeroEqzRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.czero.eqz"


@irdl_op_definition
class CZeroNezRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.czero.nez"


@irdl_op_definition
class LiRV64Op(LiOp):
    name = "riscv64.li"


@irdl_op_definition
class EcallRV64Op(EcallOp):
    name = "riscv64.ecall"


@irdl_op_definition
class LabelRV64Op(LabelOp):
    name = "riscv64.label"


@irdl_op_definition
class DirectiveRV64Op(DirectiveOp):
    name = "riscv64.directive"


@irdl_op_definition
class EbreakRV64Op(EbreakOp):
    name = "riscv64.ebreak"


@irdl_op_definition
class WfiRV64Op(WfiOp):
    name = "riscv64.wfi"


@irdl_op_definition
class CustomAssemblyInstructionRV64Op(CustomAssemblyInstructionOp):
    name = "riscv64.custom_asm"
    

# ==============================================================================
# Bit Manipulation Operations (Zb* extensions)
# ==============================================================================


@irdl_op_definition
class AndnRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.andn"
    traits = traits_def(Pure())


@irdl_op_definition
class OrnRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.orn"
    traits = traits_def(Pure())


@irdl_op_definition
class XnorRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.xnor"
    traits = traits_def(Pure())


@irdl_op_definition
class BclrRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.bclr"
    traits = traits_def(Pure())


@irdl_op_definition
class BclrIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.bclri"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BextRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.bext"
    traits = traits_def(Pure())


@irdl_op_definition
class BextIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.bexti"


@irdl_op_definition
class BinvRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.binv"
    traits = traits_def(Pure())


@irdl_op_definition
class BinvIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.binvi"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BsetRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.bset"
    traits = traits_def(Pure())


@irdl_op_definition
class BsetIRV64Op(RdRsImmShiftOperationRV64):
    name = "riscv64.bseti"

    traits = traits_def(Pure())

@irdl_op_definition
class MaxRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.max"
    traits = traits_def(Pure())


@irdl_op_definition
class MaxURV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.maxu"
    traits = traits_def(Pure())


@irdl_op_definition
class MinRV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.min"
    traits = traits_def(Pure())


@irdl_op_definition
class MinURV64Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv64.minu"
    traits = traits_def(Pure())


RISCV64 = Dialect(
    "riscv64",
    [
        AddiRV64Op,
        SltiRV64Op,
        SltiuRV64Op,
        AndiRV64Op,
        OriRV64Op,
        XoriRV64Op,
        SlliRV64Op,
        SrliRV64Op,
        SraiRV64Op,
        AddiwRV64Op,
        SraiwRV64Op,
        AddwRV64Op,
        SubwRV64Op,
        SllwRV64Op,
        SrlwRV64Op,
        SrawRV64Op,
        LuiRV64Op,
        AuipcRV64Op,
        SeqzRV64Op,
        SnezRV64Op,
        ZextBRV64Op,
        ZextWRV64Op,
        SextWRV64Op,
        AddRV64Op,
        SltRV64Op,
        SltuRV64Op,
        AndRV64Op,
        OrRV64Op,
        XorRV64Op,
        SllRV64Op,
        SrlRV64Op,
        SubRV64Op,
        SraRV64Op,
        NopRV64Op,
        JalRV64Op,
        JRV64Op,
        JalrRV64Op,
        ReturnRV64Op,
        BeqRV64Op,
        BneRV64Op,
        BltRV64Op,
        BgeRV64Op,
        BltuRV64Op,
        BgeuRV64Op,
        LbRV64Op,
        LbuRV64Op,
        LhRV64Op,
        LhuRV64Op,
        LwRV64Op,
        SbRV64Op,
        ShRV64Op,
        SwRV64Op,
        CsrrwRV64Op,
        CsrrsRV64Op,
        CsrrcRV64Op,
        CsrrwiRV64Op,
        CsrrsiRV64Op,
        CsrrciRV64Op,
        MulRV64Op,
        MulhRV64Op,
        MulhsuRV64Op,
        MulhuRV64Op,
        MulwRV64Op,
        DivRV64Op,
        DivuRV64Op,
        DivuwRV64Op,
        DivwRV64Op,
        RemRV64Op,
        RemuRV64Op,
        RemuwRV64Op,
        RemwRV64Op,
        RolRV64Op,
        RorRV64Op,
        SextHRV64Op,
        ZextHRV64Op,
        SextBRV64Op,
        RolwRV64Op,
        RorwRV64Op,
        AddUwRV64Op,
        Sh1addRV64Op,
        Sh2addRV64Op,
        Sh3addRV64Op,
        Sh1addUwRV64Op,
        Sh2addUwRV64Op,
        Sh3addUwRV64Op,
        CZeroEqzRV64Op,
        CZeroNezRV64Op,
        LiRV64Op,
        EcallRV64Op,
        LabelRV64Op,
        DirectiveRV64Op,
        EbreakRV64Op,
        WfiRV64Op,
        CustomAssemblyInstructionRV64Op,
        # Bit Manipulation Operations
        AndnRV64Op,
        OrnRV64Op,
        XnorRV64Op,
        BclrRV64Op,
        BclrIRV64Op,
        BextRV64Op,
        BextIRV64Op,
        BinvRV64Op,
        BinvIRV64Op,
        BsetRV64Op,
        BsetIRV64Op,
        MaxRV64Op,
        MaxURV64Op,
        MinRV64Op,
        MinURV64Op,
    ],
    [],
)
