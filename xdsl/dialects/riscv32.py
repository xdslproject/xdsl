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
    UImm5Attr,
    WfiOp,
    XoriOp,
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
class AddiRV32Op(AddiOp):
    name = "riscv32.addi"


@irdl_op_definition
class SltiRV32Op(SltiOp):
    name = "riscv32.slti"


@irdl_op_definition
class SltiuRV32Op(SltiuOp):
    name = "riscv32.sltiu"


@irdl_op_definition
class AndiRV32Op(AndiOp):
    name = "riscv32.andi"


@irdl_op_definition
class OriRV32Op(OriOp):
    name = "riscv32.ori"


@irdl_op_definition
class XoriRV32Op(XoriOp):
    name = "riscv32.xori"


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
class AddiwRV32Op(AddiwOp):
    name = "riscv32.addiw"

@irdl_op_definition
class SlliwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.slliw"

    traits = traits_def(Pure())

@irdl_op_definition
class SrliwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.srliw"

    traits = traits_def(Pure())

@irdl_op_definition
class SraiwRV32Op(SraiwOp):
    name = "riscv32.sraiw"


@irdl_op_definition
class AddwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.addw"
    traits = traits_def(Pure())


@irdl_op_definition
class SubwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.subw"
    traits = traits_def(Pure())


@irdl_op_definition
class SllwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sllw"
    traits = traits_def(Pure())


@irdl_op_definition
class SrlwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.srlw"
    traits = traits_def(Pure())


@irdl_op_definition
class SrawRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sraw"
    traits = traits_def(Pure())


@irdl_op_definition
class LuiRV32Op(LuiOp):
    name = "riscv32.lui"


@irdl_op_definition
class AuipcRV32Op(AuipcOp):
    name = "riscv32.auipc"


@irdl_op_definition
class SeqzRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.seqz"


@irdl_op_definition
class SnezRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.snez"


@irdl_op_definition
class ZextBRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.zext.b"
    traits = traits_def(Pure())


@irdl_op_definition
class ZextWRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.zext.w"
    traits = traits_def(Pure())


@irdl_op_definition
class SextWRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.sext.w"
    traits = traits_def(Pure())


@irdl_op_definition
class AddRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.add"
    traits = traits_def(Pure())


@irdl_op_definition
class SltRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.slt"


@irdl_op_definition
class SltuRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sltu"


@irdl_op_definition
class AndRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.and"
    traits = traits_def(Pure())


@irdl_op_definition
class OrRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.or"
    traits = traits_def(Pure())


@irdl_op_definition
class XorRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.xor"
    traits = traits_def(Pure())


@irdl_op_definition
class SllRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sll"
    traits = traits_def(Pure())


@irdl_op_definition
class SrlRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.srl"
    traits = traits_def(Pure())


@irdl_op_definition
class SubRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sub"
    traits = traits_def(Pure())


@irdl_op_definition
class SraRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sra"
    traits = traits_def(Pure())


@irdl_op_definition
class NopRV32Op(NopOp):
    name = "riscv32.nop"


@irdl_op_definition
class JalRV32Op(JalOp):
    name = "riscv32.jal"


@irdl_op_definition
class JRV32Op(JOp):
    name = "riscv32.j"


@irdl_op_definition
class JalrRV32Op(JalrOp):
    name = "riscv32.jalr"


@irdl_op_definition
class ReturnRV32Op(ReturnOp):
    name = "riscv32.ret"


@irdl_op_definition
class BeqRV32Op(BeqOp):
    name = "riscv32.beq"


@irdl_op_definition
class BneRV32Op(BneOp):
    name = "riscv32.bne"


@irdl_op_definition
class BltRV32Op(BltOp):
    name = "riscv32.blt"


@irdl_op_definition
class BgeRV32Op(BgeOp):
    name = "riscv32.bge"


@irdl_op_definition
class BltuRV32Op(BltuOp):
    name = "riscv32.bltu"


@irdl_op_definition
class BgeuRV32Op(BgeuOp):
    name = "riscv32.bgeu"


@irdl_op_definition
class LbRV32Op(LbOp):
    name = "riscv32.lb"


@irdl_op_definition
class LbuRV32Op(LbuOp):
    name = "riscv32.lbu"


@irdl_op_definition
class LhRV32Op(LhOp):
    name = "riscv32.lh"


@irdl_op_definition
class LhuRV32Op(LhuOp):
    name = "riscv32.lhu"


@irdl_op_definition
class LwRV32Op(LwOp):
    name = "riscv32.lw"


@irdl_op_definition
class SbRV32Op(SbOp):
    name = "riscv32.sb"


@irdl_op_definition
class ShRV32Op(ShOp):
    name = "riscv32.sh"


@irdl_op_definition
class SwRV32Op(SwOp):
    name = "riscv32.sw"


@irdl_op_definition
class CsrrwRV32Op(CsrrwOp):
    name = "riscv32.csrrw"


@irdl_op_definition
class CsrrsRV32Op(CsrrsOp):
    name = "riscv32.csrrs"


@irdl_op_definition
class CsrrcRV32Op(CsrrcOp):
    name = "riscv32.csrrc"


@irdl_op_definition
class CsrrwiRV32Op(CsrrwiOp):
    name = "riscv32.csrrwi"


@irdl_op_definition
class CsrrsiRV32Op(CsrrsiOp):
    name = "riscv32.csrrsi"


@irdl_op_definition
class CsrrciRV32Op(CsrrciOp):
    name = "riscv32.csrrci"


@irdl_op_definition
class MulRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.mul"
    traits = traits_def(Pure())


@irdl_op_definition
class MulhRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.mulh"
    traits = traits_def(Pure())


@irdl_op_definition
class MulhsuRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.mulhsu"
    traits = traits_def(Pure())


@irdl_op_definition
class MulhuRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.mulhu"
    traits = traits_def(Pure())


@irdl_op_definition
class MulwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.mulw"
    traits = traits_def(Pure())


@irdl_op_definition
class DivRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.div"


@irdl_op_definition
class DivuRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.divu"


@irdl_op_definition
class DivuwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.divuw"


@irdl_op_definition
class DivwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.divw"


@irdl_op_definition
class RemRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.rem"


@irdl_op_definition
class RemuRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.remu"


@irdl_op_definition
class RemuwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.remuw"


@irdl_op_definition
class RemwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.remw"


@irdl_op_definition
class RolRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.rol"
    traits = traits_def(Pure())

@irdl_op_definition
class RorRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.ror"
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
class SextHRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.sext.h"
    traits = traits_def(Pure())


@irdl_op_definition
class ZextHRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.zext.h"
    traits = traits_def(Pure())


@irdl_op_definition
class SextBRV32Op(RdRsIntegerOperation[IntRegisterType]):
    name = "riscv32.sext.b"
    traits = traits_def(Pure())


@irdl_op_definition
class RolwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.rolw"
    traits = traits_def(Pure())


@irdl_op_definition
class RorwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.rorw"
    traits = traits_def(Pure())


@irdl_op_definition
class AddUwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh1addRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sh1add"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh2addRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sh2add"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh3addRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sh3add"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh1addUwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sh1add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh2addUwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sh2add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class Sh3addUwRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.sh3add.uw"
    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.slli.uw"
    
    traits = traits_def(Pure())

@irdl_op_definition
class CZeroEqzRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.czero.eqz"


@irdl_op_definition
class CZeroNezRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.czero.nez"


@irdl_op_definition
class LiRV32Op(LiOp):
    name = "riscv32.li"


@irdl_op_definition
class EcallRV32Op(EcallOp):
    name = "riscv32.ecall"


@irdl_op_definition
class LabelRV32Op(LabelOp):
    name = "riscv32.label"


@irdl_op_definition
class DirectiveRV32Op(DirectiveOp):
    name = "riscv32.directive"


@irdl_op_definition
class EbreakRV32Op(EbreakOp):
    name = "riscv32.ebreak"


@irdl_op_definition
class WfiRV32Op(WfiOp):
    name = "riscv32.wfi"


@irdl_op_definition
class CustomAssemblyInstructionRV32Op(CustomAssemblyInstructionOp):
    name = "riscv32.custom_asm"
    

# ==============================================================================
# Bit Manipulation Operations (Zb* extensions)
# ==============================================================================


@irdl_op_definition
class AndnRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.andn"
    traits = traits_def(Pure())


@irdl_op_definition
class OrnRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.orn"
    traits = traits_def(Pure())


@irdl_op_definition
class XnorRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.xnor"
    traits = traits_def(Pure())


@irdl_op_definition
class BclrRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.bclr"
    traits = traits_def(Pure())


@irdl_op_definition
class BclrIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.bclri"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BextRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.bext"
    traits = traits_def(Pure())


@irdl_op_definition
class BextIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.bexti"


@irdl_op_definition
class BinvRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.binv"
    traits = traits_def(Pure())


@irdl_op_definition
class BinvIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.binvi"
    
    traits = traits_def(Pure())


@irdl_op_definition
class BsetRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.bset"
    traits = traits_def(Pure())


@irdl_op_definition
class BsetIRV32Op(RdRsImmShiftOperationRV32):
    name = "riscv32.bseti"

    traits = traits_def(Pure())

@irdl_op_definition
class MaxRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.max"
    traits = traits_def(Pure())


@irdl_op_definition
class MaxURV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.maxu"
    traits = traits_def(Pure())


@irdl_op_definition
class MinRV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.min"
    traits = traits_def(Pure())


@irdl_op_definition
class MinURV32Op(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    name = "riscv32.minu"
    traits = traits_def(Pure())


RISCV32 = Dialect(
    "riscv32",
    [
        AddiRV32Op,
        SltiRV32Op,
        SltiuRV32Op,
        AndiRV32Op,
        OriRV32Op,
        XoriRV32Op,
        SlliRV32Op,
        SrliRV32Op,
        SraiRV32Op,
        AddiwRV32Op,
        SraiwRV32Op,
        AddwRV32Op,
        SubwRV32Op,
        SllwRV32Op,
        SrlwRV32Op,
        SrawRV32Op,
        LuiRV32Op,
        AuipcRV32Op,
        SeqzRV32Op,
        SnezRV32Op,
        ZextBRV32Op,
        ZextWRV32Op,
        SextWRV32Op,
        AddRV32Op,
        SltRV32Op,
        SltuRV32Op,
        AndRV32Op,
        OrRV32Op,
        XorRV32Op,
        SllRV32Op,
        SrlRV32Op,
        SubRV32Op,
        SraRV32Op,
        NopRV32Op,
        JalRV32Op,
        JRV32Op,
        JalrRV32Op,
        ReturnRV32Op,
        BeqRV32Op,
        BneRV32Op,
        BltRV32Op,
        BgeRV32Op,
        BltuRV32Op,
        BgeuRV32Op,
        LbRV32Op,
        LbuRV32Op,
        LhRV32Op,
        LhuRV32Op,
        LwRV32Op,
        SbRV32Op,
        ShRV32Op,
        SwRV32Op,
        CsrrwRV32Op,
        CsrrsRV32Op,
        CsrrcRV32Op,
        CsrrwiRV32Op,
        CsrrsiRV32Op,
        CsrrciRV32Op,
        MulRV32Op,
        MulhRV32Op,
        MulhsuRV32Op,
        MulhuRV32Op,
        MulwRV32Op,
        DivRV32Op,
        DivuRV32Op,
        DivuwRV32Op,
        DivwRV32Op,
        RemRV32Op,
        RemuRV32Op,
        RemuwRV32Op,
        RemwRV32Op,
        RolRV32Op,
        RorRV32Op,
        SextHRV32Op,
        ZextHRV32Op,
        SextBRV32Op,
        RolwRV32Op,
        RorwRV32Op,
        AddUwRV32Op,
        Sh1addRV32Op,
        Sh2addRV32Op,
        Sh3addRV32Op,
        Sh1addUwRV32Op,
        Sh2addUwRV32Op,
        Sh3addUwRV32Op,
        CZeroEqzRV32Op,
        CZeroNezRV32Op,
        LiRV32Op,
        EcallRV32Op,
        LabelRV32Op,
        DirectiveRV32Op,
        EbreakRV32Op,
        WfiRV32Op,
        CustomAssemblyInstructionRV32Op,
        # Bit Manipulation Operations
        AndnRV32Op,
        OrnRV32Op,
        XnorRV32Op,
        BclrRV32Op,
        BclrIRV32Op,
        BextRV32Op,
        BextIRV32Op,
        BinvRV32Op,
        BinvIRV32Op,
        BsetRV32Op,
        BsetIRV32Op,
        MaxRV32Op,
        MaxURV32Op,
        MinRV32Op,
        MinURV32Op,
    ],
    [],
)
