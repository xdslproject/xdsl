from __future__ import annotations
from dataclasses import dataclass, field

from typing import Union, TypeVar, Type

from xdsl.ir import Dialect, MLContext, Operation, Data
from xdsl.irdl import irdl_op_definition, irdl_attr_definition, builder, AnyOf
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import StringAttr, IntegerAttr, OpAttr


@dataclass(frozen=True)
class Register:
    """A riscv register."""

    index: int = field()
    """The register index. Can be between 0 and 31."""

    abi_names = {
        "zero": 0,
        "ra": 1,
        "sp": 2,
        "gp": 3,
        "tp": 4,
        "t0": 5,
        "t1": 6,
        "t2": 7,
        "fp": 8,
        "s0": 8,
        "s1": 9,
        "a0": 10,
        "a1": 11,
        "a2": 12,
        "a3": 13,
        "a4": 14,
        "a5": 15,
        "a6": 16,
        "a7": 17,
        "s2": 18,
        "s3": 19,
        "s4": 20,
        "s5": 21,
        "s6": 22,
        "s7": 23,
        "s8": 24,
        "s9": 25,
        "s10": 26,
        "s11": 27,
        "t3": 28,
        "t4": 29,
        "t5": 30,
        "t6": 31
    }

    @staticmethod
    def from_index(index: int) -> Register:
        assert 32 > index >= 0
        register = Register(index)
        return register

    @staticmethod
    def from_name(name: str) -> Register:
        if name in Register.abi_names:
            return Register.from_index(Register.abi_names[name])
        if name[0] == 'x' and name[1:].isnumeric():
            return Register.from_index(int(name[1:]))
        assert False, "register with unknown name"

    def get_abi_name(self) -> str:
        for name, index in Register.abi_names.items():
            if index == self.index:
                return name
        assert False, "Register with unknown index"


@irdl_attr_definition
class RegisterAttr(Data[Register]):
    name = "riscv.reg"

    @staticmethod
    def parse_parameter(parser: Parser) -> Register:
        name = parser.parse_while(lambda x: x != ">")
        return Register.from_name(name)

    @staticmethod
    def print_parameter(data: Register, printer: Printer) -> None:
        printer.print_string(data)

    @staticmethod
    @builder
    def from_index(index: int) -> RegisterAttr:
        return RegisterAttr(Register.from_index(index))

    @staticmethod
    @builder
    def from_name(name: str) -> RegisterAttr:
        return RegisterAttr(Register.from_name(name))

    @staticmethod
    @builder
    def from_register(register: Register) -> RegisterAttr:
        return RegisterAttr(register)


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "riscv.label"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        data = parser.parse_while(lambda x: x != ">")
        return data

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        printer.print_string(data)

    @staticmethod
    @builder
    def from_str(name: str) -> LabelAttr:
        return LabelAttr(name)


Op = TypeVar("Op", bound=Operation)


class Riscv1Rd1Rs1ImmOperation(Operation):
    rd: OpAttr[RegisterAttr]
    rs1: OpAttr[RegisterAttr]
    immediate: OpAttr[IntegerAttr]

    @classmethod
    def get(cls: Type[Op], rd, rs1, immediate) -> Op:
        if isinstance(immediate, int):
            immediate = IntegerAttr.from_int_and_width(immediate, 64)
        return cls.build(attributes={
            "rd": rd,
            "rs1": rs1,
            "immediate": immediate
        })


class Riscv1Rd1Rs1OffOperation(Operation):
    rd: OpAttr[RegisterAttr]
    rs1: OpAttr[RegisterAttr]
    offset: OpAttr[AnyOf([IntegerAttr, LabelAttr])]

    @classmethod
    def get(cls: Type[Op], rd, rs1, offset) -> Op:
        if isinstance(offset, int):
            offset = IntegerAttr.from_int_and_width(offset, 64)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)
        return cls.build(attributes={"rd": rd, "rs1": rs1, "offset": offset})


class Riscv2Rs1ImmOperation(Operation):
    rs1: OpAttr[RegisterAttr]
    rs2: OpAttr[RegisterAttr]
    immediate: OpAttr[IntegerAttr]

    @classmethod
    def get(cls: Type[Op], rs1, rs2, immediate) -> Op:
        if isinstance(immediate, int):
            immediate = IntegerAttr.from_int_and_width(immediate, 64)
        return cls.build(attributes={
            "rs1": rs1,
            "rs2": rs2,
            "immediate": immediate
        })


class Riscv2Rs1OffOperation(Operation):
    rs1: OpAttr[RegisterAttr]
    rs2: OpAttr[RegisterAttr]
    offset: OpAttr[AnyOf([IntegerAttr, LabelAttr])]

    @classmethod
    def get(cls: Type[Op], rs1, rs2, offset) -> Op:
        if isinstance(offset, int):
            offset = IntegerAttr.from_int_and_width(offset, 64)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)
        return cls.build(attributes={"rs1": rs1, "rs2": rs2, "offset": offset})


class Riscv1Rd2RsOperation(Operation):
    rd: OpAttr[RegisterAttr]
    rs1: OpAttr[RegisterAttr]
    rs2: OpAttr[RegisterAttr]

    @classmethod
    def get(cls: Type[Op], rd, rs1, rs2) -> Op:
        return cls.build(attributes={"rd": rd, "rs1": rs1, "rs2": rs2})


class Riscv1Rs1Rt1OffOperation(Operation):
    rs: OpAttr[RegisterAttr]
    rt: OpAttr[RegisterAttr]
    offset: OpAttr[AnyOf([IntegerAttr, LabelAttr])]

    @classmethod
    def get(cls: Type[Op], rs, rt, offset) -> Op:
        if isinstance(offset, int):
            offset = IntegerAttr.from_int_and_width(offset, 64)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)
        return cls.build(attributes={"rs": rs, "rt": rt, "offset": offset})


class Riscv1Rd1ImmOperation(Operation):
    rd: OpAttr[RegisterAttr]
    immediate: OpAttr[IntegerAttr]

    @classmethod
    def get(cls: Type[Op], rd, immediate) -> Op:
        if isinstance(immediate, int):
            immediate = IntegerAttr.from_int_and_width(immediate, 64)
        return cls.build(attributes={"rd": rd, "immediate": immediate})


class Riscv1Rd1OffOperation(Operation):
    rd: OpAttr[RegisterAttr]
    offset: OpAttr[AnyOf([IntegerAttr, LabelAttr])]

    @classmethod
    def get(cls: Type[Op], rd, offset) -> Op:
        if isinstance(offset, int):
            offset = IntegerAttr.from_int_and_width(offset, 64)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)
        return cls.build(attributes={"rd": rd, "offset": offset})


class Riscv1Rs1OffOperation(Operation):
    rs: OpAttr[RegisterAttr]
    offset: OpAttr[AnyOf([IntegerAttr, LabelAttr])]

    @classmethod
    def get(cls: Type[Op], rs, offset) -> Op:
        if isinstance(offset, int):
            offset = IntegerAttr.from_int_and_width(offset, 64)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)
        return cls.build(attributes={"rs": rs, "offset": offset})


class Riscv1Rd1RsOperation(Operation):
    rd: OpAttr[RegisterAttr]
    rs: OpAttr[RegisterAttr]

    @classmethod
    def get(cls: Type[Op], rd, rs) -> Op:
        return cls.build(attributes={"rd": rd, "rs": rs})


class RiscvNoParamsOperation(Operation):

    @classmethod
    def get(cls: Type[Op]) -> Op:
        return cls.build()


# Loads


@irdl_op_definition
class LBOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.lb"


@irdl_op_definition
class LBUOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.lbu"


@irdl_op_definition
class LHOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.lh"


@irdl_op_definition
class LHUOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.lhu"


@irdl_op_definition
class LWOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.lw"


# Stores


@irdl_op_definition
class SBOp(Riscv2Rs1ImmOperation):
    name = "riscv.sb"


@irdl_op_definition
class SHOp(Riscv2Rs1ImmOperation):
    name = "riscv.sh"


@irdl_op_definition
class SWOp(Riscv2Rs1ImmOperation):
    name = "riscv.sw"


# Branches


@irdl_op_definition
class BEQOp(Riscv2Rs1OffOperation):
    name = "riscv.beq"


@irdl_op_definition
class BNEOp(Riscv2Rs1OffOperation):
    name = "riscv.bne"


@irdl_op_definition
class BLTOp(Riscv2Rs1OffOperation):
    name = "riscv.blt"


@irdl_op_definition
class BGEOp(Riscv2Rs1OffOperation):
    name = "riscv.bge"


@irdl_op_definition
class BLTUOp(Riscv2Rs1OffOperation):
    name = "riscv.bltu"


@irdl_op_definition
class BGEUOp(Riscv2Rs1OffOperation):
    name = "riscv.bgeu"


# Shifts


@irdl_op_definition
class SLLOp(Riscv1Rd2RsOperation):
    name = "riscv.sll"


@irdl_op_definition
class SLLIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.slli"


@irdl_op_definition
class SRLOp(Riscv1Rd2RsOperation):
    name = "riscv.srl"


@irdl_op_definition
class SRLIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.srli"


@irdl_op_definition
class SRAOp(Riscv1Rd2RsOperation):
    name = "riscv.sra"


@irdl_op_definition
class SRAIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.srai"


# Arithmetic


@irdl_op_definition
class AddOp(Riscv1Rd2RsOperation):
    name = "riscv.add"


@irdl_op_definition
class AddIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.addi"


@irdl_op_definition
class SubOp(Riscv1Rd2RsOperation):
    name = "riscv.sub"


@irdl_op_definition
class LUIOp(Riscv1Rd1ImmOperation):
    name = "riscv.lui"


@irdl_op_definition
class AUIPCOp(Riscv1Rd1ImmOperation):
    name = "riscv.auipc"


# Logical


@irdl_op_definition
class XOROp(Riscv1Rd2RsOperation):
    name = "riscv.xor"


@irdl_op_definition
class XORIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.xori"


@irdl_op_definition
class OROp(Riscv1Rd2RsOperation):
    name = "riscv.or"


@irdl_op_definition
class ORIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.ori"


@irdl_op_definition
class ANDOp(Riscv1Rd2RsOperation):
    name = "riscv.and"


@irdl_op_definition
class ANDIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.andi"


# Compare


@irdl_op_definition
class SLTOp(Riscv1Rd2RsOperation):
    name = "riscv.slt"


@irdl_op_definition
class SLTIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.slti"


@irdl_op_definition
class SLTUOp(Riscv1Rd2RsOperation):
    name = "riscv.sltu"


@irdl_op_definition
class SLTUIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv.sltui"


# Jump & Link


@irdl_op_definition
class JALOp(Riscv1Rd1OffOperation):
    name = "riscv.jal"


@irdl_op_definition
class JALROp(Riscv1Rd1Rs1OffOperation):
    name = "riscv.jalr"


# System


@irdl_op_definition
class SCALLOp(RiscvNoParamsOperation):
    name = "riscv.scall"


@irdl_op_definition
class SBREAKOp(RiscvNoParamsOperation):
    name = "riscv.sbreak"


#  Optional Multiply-Devide Instruction Extension (RVM)


@irdl_op_definition
class MULOp(Riscv1Rd2RsOperation):
    name = "riscv.mul"


@irdl_op_definition
class MULHOp(Riscv1Rd2RsOperation):
    name = "riscv.mulh"


@irdl_op_definition
class MULHSUOp(Riscv1Rd2RsOperation):
    name = "riscv.mulhsu"


@irdl_op_definition
class MULHUOp(Riscv1Rd2RsOperation):
    name = "riscv.mulhu"


@irdl_op_definition
class DIVOp(Riscv1Rd2RsOperation):
    name = "riscv.div"


@irdl_op_definition
class DIVUOp(Riscv1Rd2RsOperation):
    name = "riscv.divu"


@irdl_op_definition
class REMOp(Riscv1Rd2RsOperation):
    name = "riscv.rem"


@irdl_op_definition
class REMUOp(Riscv1Rd2RsOperation):
    name = "riscv.remu"


# Pseudo Ops (Table 25.2)
#
# We do not include sext.w as we cannot parse this operation yet
# and also do not include the floating point operations.


@irdl_op_definition
class NOPOp(RiscvNoParamsOperation):
    name = "riscv.nop"


@irdl_op_definition
class LIOp(Riscv1Rd1ImmOperation):
    name = "riscv.li"


@irdl_op_definition
class MVOp(Riscv1Rd1RsOperation):
    name = "riscv.mv"


@irdl_op_definition
class NOTOp(Riscv1Rd1RsOperation):
    name = "riscv.not"


@irdl_op_definition
class NEGOp(Riscv1Rd1RsOperation):
    name = "riscv.neg"


@irdl_op_definition
class NEGWOp(Riscv1Rd1RsOperation):
    name = "riscv.negw"


@irdl_op_definition
class SEQZOp(Riscv1Rd1RsOperation):
    name = "riscv.seqz"


@irdl_op_definition
class SNEZOp(Riscv1Rd1RsOperation):
    name = "riscv.snez"


@irdl_op_definition
class SLTZOp(Riscv1Rd1RsOperation):
    name = "riscv.sltz"


@irdl_op_definition
class SGTZOp(Riscv1Rd1RsOperation):
    name = "riscv.sgtz"


@irdl_op_definition
class BEQZOp(Riscv1Rs1OffOperation):
    name = "riscv.beqz"


@irdl_op_definition
class BNEZOp(Riscv1Rs1OffOperation):
    name = "riscv.bnez"


@irdl_op_definition
class BLEZOp(Riscv1Rs1OffOperation):
    name = "riscv.blez"


@irdl_op_definition
class BGEZOp(Riscv1Rs1OffOperation):
    name = "riscv.bgez"


@irdl_op_definition
class BLTZOp(Riscv1Rs1OffOperation):
    name = "riscv.bltz"


@irdl_op_definition
class BGTZOp(Riscv1Rs1OffOperation):
    name = "riscv.bgtz"


@irdl_op_definition
class BGTOp(Riscv1Rs1Rt1OffOperation):
    name = "riscv.bgt"


@irdl_op_definition
class BLEOp(Riscv1Rs1Rt1OffOperation):
    name = "riscv.ble"


@irdl_op_definition
class BGTUOp(Riscv1Rs1Rt1OffOperation):
    name = "riscv.bgtu"


@irdl_op_definition
class BLEUOp(Riscv1Rs1Rt1OffOperation):
    name = "riscv.bleu"


@irdl_op_definition
class RETOp(RiscvNoParamsOperation):
    name = "riscv.ret"


# Auxiliary operations


@irdl_op_definition
class LabelOp(Operation):
    name = "riscv.label"
    label: OpAttr[LabelAttr]

    @staticmethod
    def get(label: Union[str, LabelAttr]) -> LabelOp:
        return LabelOp.build(attributes={"label": label})


@irdl_op_definition
class DirectiveOp(Operation):
    name = "riscv.directive"
    directive: OpAttr[StringAttr]
    value: OpAttr[StringAttr]

    @staticmethod
    def get(directive: Union[str, StringAttr],
            value: Union[str, StringAttr] = "") -> DirectiveOp:
        return DirectiveOp.build(attributes={
            "directive": directive,
            "value": value
        })


RISCV = Dialect(
    [
        # Loads
        LBOp,
        LBUOp,
        LHOp,
        LHUOp,
        LWOp,

        # Stores
        SBOp,
        SHOp,
        SWOp,

        # Branches
        BEQOp,
        BNEOp,
        BLTOp,
        BGEOp,
        BLTUOp,
        BGEUOp,

        # Shifts
        SLLOp,
        SLLIOp,
        SRLOp,
        SRLIOp,
        SRAOp,
        SRAIOp,

        # Arithmetic
        AddOp,
        AddIOp,
        SubOp,
        LUIOp,
        AUIPCOp,

        # Logical
        XOROp,
        XORIOp,
        OROp,
        ORIOp,
        ANDOp,
        ANDIOp,

        # Compare
        SLTOp,
        SLTIOp,
        SLTUOp,
        SLTUIOp,

        # Jump & Link
        JALOp,
        JALROp,

        # System
        SCALLOp,
        SBREAKOp,

        #  Optional Multiply-Devide Instruction Extension (RVM)
        MULOp,
        MULHOp,
        MULHSUOp,
        MULHUOp,
        DIVOp,
        DIVUOp,
        REMOp,
        REMUOp,

        # Pseudo Ops
        NOPOp,
        LIOp,
        MVOp,
        NOTOp,
        NEGOp,
        NEGWOp,
        SEQZOp,
        SNEZOp,
        SLTZOp,
        SGTZOp,
        BEQZOp,
        BNEZOp,
        BLEZOp,
        BGEZOp,
        BLTZOp,
        BGTZOp,
        BGTOp,
        BLEOp,
        BGTUOp,
        BLEUOp,
        RETOp,

        # Auxiliary operations
        LabelOp,
        DirectiveOp
    ],
    [RegisterAttr, LabelAttr])