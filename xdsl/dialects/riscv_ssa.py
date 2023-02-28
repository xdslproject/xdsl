from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Type, Dict, Union, Optional, Any, List, TypeVar, Annotated

from xdsl.dialects.builtin import StringAttr, IntegerAttr, AnyIntegerAttr, IntAttr
from xdsl.ir import (Operation, ParametrizedAttribute, SSAValue, Dialect,
                     Attribute, Data, OpResult, Region)
from xdsl.irdl import (irdl_op_definition, irdl_attr_definition, OptOpResult,
                       VarOperand, SingleBlockRegion, OpAttr, OptOpAttr,
                       OptOperand, Operand, ParameterDef)
from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


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
        assert False and "register with unknown name"

    def get_abi_name(self) -> str:
        for name, index in Register.abi_names.items():
            if index == self.index:
                return name
        assert False and "Register with unknown index"


@irdl_attr_definition
class RegisterAttr(Data[Register]):
    name = "riscv.reg"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> Register:
        token = parser.tokenizer.next_token()
        if token.text not in Register.abi_names:
            parser.raise_error("Unknown RISC-V Register {}".format(token.text),
                               token)
        return Register.from_name(token.text)

    @staticmethod
    def print_parameter(data: Register, printer: Printer) -> None:
        printer.print_string(data.get_abi_name())

    @staticmethod
    def from_index(index: int) -> RegisterAttr:
        return RegisterAttr(data=Register.from_index(index))

    @staticmethod
    def from_name(name: str) -> RegisterAttr:
        return RegisterAttr(data=Register.from_name(name))

    @staticmethod
    def from_register(register: Register) -> RegisterAttr:
        return RegisterAttr(data=register)


@irdl_attr_definition
class LabelAttr(ParametrizedAttribute):
    name = "riscv.label"
    label: ParameterDef[StringAttr]

    def string_value(self):
        return self.label.data

    @staticmethod
    def from_str(name: str) -> LabelAttr:
        return LabelAttr([StringAttr.from_str(name)])


Op = TypeVar("Op", bound=Operation)


@irdl_attr_definition
class RegisterType(ParametrizedAttribute):
    name = "riscv_ssa.reg"


class Riscv1Rd1Rs1ImmOperation(Operation, ABC):
    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    immediate: OpAttr[IntAttr | LabelAttr]
    comment: OptOpAttr[StringAttr]

    @classmethod
    def get(cls: Type[Op],
            rs1: Union[Operation, SSAValue],
            immediate: Union[int, IntAttr, str, LabelAttr],
            comment: Optional[str] = None) -> Op:
        if isinstance(immediate, int):
            immediate = IntAttr.from_int(immediate)
        elif isinstance(immediate, str):
            immediate = LabelAttr.from_str(immediate)

        attributes: Dict[str, Attribute] = {
            "immediate": immediate,
        }
        if comment:
            attributes["comment"] = StringAttr.from_str(comment)
        return cls.build(operands=[rs1],
                         result_types=[RegisterType()],
                         attributes=attributes)


class Riscv2Rs1ImmOperation(Operation, ABC):
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    immediate: OpAttr[IntAttr | LabelAttr]
    comment: OptOpAttr[StringAttr]

    @classmethod
    def get(cls: Type[Op],
            rs1: Union[Operation, SSAValue],
            rs2: Union[Operation, SSAValue],
            immediate: Union[int, IntAttr, str, LabelAttr],
            comment: Optional[str] = None) -> Op:
        if isinstance(immediate, int):
            immediate = IntAttr.from_int(immediate)
        elif isinstance(immediate, str):
            immediate = LabelAttr.from_str(immediate)

        attributes: Dict[str, Any] = {
            "immediate": immediate,
        }
        if comment:
            attributes["comment"] = comment
        return cls.build(operands=[rs1, rs2], attributes=attributes)


class Riscv2Rs1OffOperation(Operation, ABC):
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    offset: OpAttr[IntAttr | LabelAttr]
    comment: OptOpAttr[StringAttr]

    @classmethod
    def get(cls: Type[Op],
            rs1: Union[Operation, SSAValue],
            rs2: Union[Operation, SSAValue],
            offset: Union[int, IntAttr, LabelAttr, str],
            comment: Optional[str] = None) -> Op:
        if isinstance(offset, int):
            offset = IntAttr.from_int(offset)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)

        attributes: Dict[str, Attribute] = {
            "offset": offset,
        }
        if comment:
            attributes["comment"] = StringAttr.from_str(comment)
        return cls.build(operands=[rs1, rs2], attributes=attributes)


class Riscv1Rd2RsOperation(Operation, ABC):
    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    comment: OptOpAttr[StringAttr]

    @classmethod
    def get(cls,
            rs1: Union[Operation, SSAValue],
            rs2: Union[Operation, SSAValue],
            comment: Optional[str] = None) -> Riscv1Rd2RsOperation:

        attributes: Dict[str, Any] = {}
        if comment:
            attributes["comment"] = StringAttr.from_str(comment)
        return cls.build(operands=[rs1, rs2],
                         attributes=attributes,
                         result_types=[RegisterType()])


class Riscv1OffOperation(Operation, ABC):
    offset: OpAttr[IntAttr | LabelAttr]
    comment: OptOpAttr[StringAttr]

    @classmethod
    def get(cls: Type[Op],
            offset: Union[int, IntAttr, LabelAttr, str],
            comment: Optional[str] = None) -> Op:
        if isinstance(offset, int):
            offset = IntAttr.from_int(offset)
        if isinstance(offset, str):
            offset = LabelAttr.from_str(offset)

        attributes: Dict[str, Any] = {
            "offset": offset,
        }
        if comment:
            attributes["comment"] = comment
        return cls.build(attributes=attributes)


class Riscv1Rd1ImmOperation(Operation, ABC):
    rd: Annotated[OpResult, RegisterType]
    immediate: OpAttr[IntAttr | LabelAttr]
    comment: OptOpAttr[StringAttr]

    @classmethod
    def get(cls,
            immediate: int | IntAttr | str | LabelAttr,
            comment: Optional[str] = None) -> Riscv1Rd1ImmOperation:
        if isinstance(immediate, int):
            immediate = IntAttr.from_int(immediate)
        if isinstance(immediate, str):
            immediate = LabelAttr.from_str(immediate)

        attributes: Dict[str, Any] = {
            "immediate": immediate,
        }
        if comment:
            attributes["comment"] = comment
        return cls.build(result_types=[RegisterType()], attributes=attributes)


@irdl_op_definition
class LBOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.lb"


@irdl_op_definition
class LBUOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.lbu"


@irdl_op_definition
class LHOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.lh"


@irdl_op_definition
class LHUOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.lhu"


@irdl_op_definition
class LWOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.lw"


# Stores


@irdl_op_definition
class SBOp(Riscv2Rs1ImmOperation):
    name = "riscv_ssa.sb"


@irdl_op_definition
class SHOp(Riscv2Rs1ImmOperation):
    name = "riscv_ssa.sh"


@irdl_op_definition
class SWOp(Riscv2Rs1ImmOperation):
    name = "riscv_ssa.sw"


# Branches


@irdl_op_definition
class BEQOp(Riscv2Rs1OffOperation):
    name = "riscv_ssa.beq"


@irdl_op_definition
class BNEOp(Riscv2Rs1OffOperation):
    name = "riscv_ssa.bne"


@irdl_op_definition
class BLTOp(Riscv2Rs1OffOperation):
    name = "riscv_ssa.blt"


@irdl_op_definition
class BGEOp(Riscv2Rs1OffOperation):
    name = "riscv_ssa.bge"


@irdl_op_definition
class BLTUOp(Riscv2Rs1OffOperation):
    name = "riscv_ssa.bltu"


@irdl_op_definition
class BGEUOp(Riscv2Rs1OffOperation):
    name = "riscv_ssa.bgeu"


# Shifts


@irdl_op_definition
class SLLOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.sll"


@irdl_op_definition
class SLLIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.slli"


@irdl_op_definition
class SRLOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.srl"


@irdl_op_definition
class SRLIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.srli"


@irdl_op_definition
class SRAOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.sra"


@irdl_op_definition
class SRAIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.srai"


# Arithmetic


@irdl_op_definition
class AddOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.add"


@irdl_op_definition
class AddIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.addi"


@irdl_op_definition
class SubOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.sub"


@irdl_op_definition
class LUIOp(Riscv1Rd1ImmOperation):
    name = "riscv_ssa.lui"


@irdl_op_definition
class LIOp(Riscv1Rd1ImmOperation):
    name = "riscv_ssa.li"


@irdl_op_definition
class AUIPCOp(Riscv1Rd1ImmOperation):
    name = "riscv_ssa.auipc"


# Logical


@irdl_op_definition
class XOROp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.xor"


@irdl_op_definition
class XORIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.xori"


@irdl_op_definition
class OROp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.or"


@irdl_op_definition
class ORIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.ori"


@irdl_op_definition
class ANDOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.and"


@irdl_op_definition
class ANDIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.andi"


# Compare


@irdl_op_definition
class SLTOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.slt"


@irdl_op_definition
class SLTIOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.slti"


@irdl_op_definition
class SLTUOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.sltu"


@irdl_op_definition
class SLTIUOp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.sltiu"


# Jump & Link


@irdl_op_definition
class JOp(Riscv1OffOperation):
    name = "riscv_ssa.j"


@irdl_op_definition
class JALOp(Riscv1Rd1ImmOperation):
    name = "riscv_ssa.jal"


@irdl_op_definition
class JALROp(Riscv1Rd1Rs1ImmOperation):
    name = "riscv_ssa.jalr"


# System


@irdl_op_definition
class ECALLOp(Operation):
    name = "riscv_ssa.ecall"
    args: Annotated[VarOperand, RegisterType]
    syscall_num: OpAttr[AnyIntegerAttr]
    result: Annotated[OptOpResult, RegisterType]
    """
    Some syscalls return values by putting them into a0. The result register will represent a0.
    """

    @classmethod
    def get(cls,
            num: int | AnyIntegerAttr,
            has_result: bool = False,
            operands: list[SSAValue | Operation] | None = None):
        if operands is None:
            operands = []
        if isinstance(num, int):
            num = IntegerAttr.from_int_and_width(num, 32)
        return cls.build(
            operands=[operands],
            attributes={'syscall_num': num},
            result_types=[[RegisterType()]] if has_result else [[]])

    def verify_(self):
        assert len(self.args) < 7


@irdl_op_definition
class EBREAKOp(Operation):
    name = "riscv_ssa.ebreak"


#  Optional Multiply-Divide Instruction Extension (RVM)


@irdl_op_definition
class MULOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.mul"


@irdl_op_definition
class MULHOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.mulh"


@irdl_op_definition
class MULHSUOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.mulhsu"


@irdl_op_definition
class MULHUOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.mulhu"


@irdl_op_definition
class DIVOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.div"


@irdl_op_definition
class DIVUOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.divu"


@irdl_op_definition
class REMOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.rem"


@irdl_op_definition
class REMUOp(Riscv1Rd2RsOperation):
    name = "riscv_ssa.remu"


@irdl_op_definition
class CallOp(Operation):
    name = "riscv_ssa.call"
    args: Annotated[VarOperand, RegisterType]
    func_name: OpAttr[StringAttr]
    result: Annotated[OptOpResult, RegisterType]

    @classmethod
    def get(cls: Type[Op],
            func_name: Union[str, StringAttr],
            args: List[Union[Operation, SSAValue]],
            has_result: bool = True,
            comment: Optional[Union[str, StringAttr]] = None) -> Op:
        attributes: Dict[str, Any] = {"func_name": func_name}
        if comment is not None:
            attributes["comment"] = StringAttr.build(comment)
        return cls.build(
            operands=[args],
            result_types=[[RegisterType()]] if has_result else [[]],
            attributes=attributes)


@irdl_op_definition
class LabelOp(Operation):
    name = "riscv_ssa.label"
    label: OpAttr[LabelAttr]

    @classmethod
    def get(cls: Type[Op],
            label: Union[str, StringAttr],
            comment: Optional[str] = None) -> Op:
        attributes: Dict[str, Any] = {
            "label": label,
        }
        if comment:
            attributes["comment"] = comment
        return cls.build(operands=[], result_types=[], attributes=attributes)


@irdl_op_definition
class DirectiveOp(Operation):
    name = "riscv_ssa.directive"
    directive: OpAttr[StringAttr]
    value: OpAttr[StringAttr]

    @classmethod
    def get(cls: Type[Op], directive: str | StringAttr,
            value: str | StringAttr) -> Op:
        attributes: Dict[str, Any] = {
            "directive":
            directive if isinstance(directive, StringAttr) else
            StringAttr.from_str(directive),
            "value":
            value
            if isinstance(value, StringAttr) else StringAttr.from_str(value),
        }
        return cls.build(operands=[], result_types=[], attributes=attributes)


@irdl_op_definition
class FuncOp(Operation):
    name = "riscv_ssa.func"

    func_name: OpAttr[StringAttr]
    func_body: SingleBlockRegion

    @staticmethod
    def from_region(name: str, region: Region) -> FuncOp:
        attributes: dict[str, Attribute] = {
            "func_name": StringAttr.from_str(name)
        }

        return FuncOp.create(attributes=attributes, regions=[region])

    @staticmethod
    def from_ops(name: str, ops: list[Operation]) -> FuncOp:
        region = Region.from_operation_list(ops)
        return FuncOp.from_region(name, region)

    def verify_(self):
        # Check that the returned value matches the type of the function
        if len(self.func_body.blocks) != 1:
            raise VerifyException("Expected FuncOp to contain one block")

        block = self.func_body.blocks[0]

        if not len(block.ops):
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.ops[-1]

        if not isinstance(last_op, ReturnOp):
            raise VerifyException(
                "Expected last op of FuncOp to be a ReturnOp")


@irdl_op_definition
class ReturnOp(Operation):
    name = "riscv_ssa.return"
    value: Annotated[OptOperand, RegisterType]

    @classmethod
    def get(cls: Type[Op],
            value: Optional[Union[Operation, SSAValue]] = None) -> Op:
        operands: list[list[Operation
                            | SSAValue]] = [[]] if value is None else [[value]]
        return cls.build(operands=operands)


# debugging instructions:


@irdl_op_definition
class SectionOp(Operation):
    """
    This instruction corresponds to a section. Its block can be added to during
    the lowering process.
    """
    name = 'riscv.section'

    directive: OpAttr[StringAttr]
    data: SingleBlockRegion

    @staticmethod
    def from_region(directive: str | StringAttr, region: Region) -> SectionOp:
        if isinstance(directive, str):
            directive = StringAttr.from_str(directive)
        return SectionOp.create(attributes={'directive': directive},
                                regions=[region])

    @staticmethod
    def from_ops(directive: str | StringAttr,
                 ops: list[Operation]) -> SectionOp:
        region = Region.from_operation_list(ops)
        return SectionOp.from_region(directive, region)


riscv_ssa_attrs: List[Type[Attribute]] = [RegisterType]
riscv_ssa_ops: List[Type[Operation]] = [
    LBOp, LBUOp, LHOp, LHUOp, LWOp, SBOp, SHOp, SWOp, BEQOp, BNEOp, BLTOp,
    BGEOp, BLTUOp, BGEUOp, AddOp, AddIOp, SubOp, LUIOp, LIOp, AUIPCOp, XOROp,
    XORIOp, OROp, ORIOp, ANDOp, ANDIOp, SLTOp, SLTIOp, SLTUOp, SLTIUOp, JOp,
    JALOp, JALROp, ECALLOp, EBREAKOp, MULOp, MULHOp, MULHSUOp, MULHUOp, DIVOp,
    DIVUOp, REMOp, REMUOp, LabelOp, CallOp, FuncOp, ReturnOp, SectionOp
]
RISCVSSA = Dialect(riscv_ssa_ops, riscv_ssa_attrs)
