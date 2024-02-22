from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from io import StringIO
from typing import IO, ClassVar, Generic, TypeAlias, TypeVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    Operation,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_result_def,
    result_def,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


class X86RegisterType(Data[str], TypeAttribute, ABC):
    """
    An x86 register type.
    """

    _unallocated: ClassVar[Self | None] = None

    @classmethod
    def unallocated(cls) -> Self:
        if cls._unallocated is None:
            cls._unallocated = cls("")
        return cls._unallocated

    @property
    def register_name(self) -> str:
        """Returns name if allocated, raises ValueError if not"""
        if not self.is_allocated:
            raise ValueError("Cannot get name for unallocated register")
        return self.data

    @property
    def is_allocated(self) -> bool:
        """Returns true if an x86 register is allocated, otherwise false"""
        return bool(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            name = parser.parse_optional_identifier()
            if name is None:
                return ""
            if not name.startswith("e") and not name.startswith("r"):
                assert name in cls.abi_index_by_name(), f"{name}"
            return name

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data)

    def verify(self) -> None:
        name = self.data
        if not self.is_allocated or name.startswith("e") or name.startswith("r"):
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")

    @classmethod
    @abstractmethod
    def instruction_set_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        raise NotImplementedError()


@irdl_attr_definition
class GeneralRegisterType(X86RegisterType):
    """
    An x86 register type.
    """

    name = "x86.reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return GeneralRegisterType.X86_INDEX_BY_NAME

    X86_INDEX_BY_NAME = {
        "rax": 0,
        "rcx": 1,
        "rdx": 2,
        "rbx": 3,
        "rsp": 4,
        "rbp": 5,
        "rsi": 6,
        "rdi": 7,
        "r8": 8,
        "r9": 9,
        "r10": 10,
        "r11": 11,
        "r12": 12,
        "r13": 13,
        "r14": 14,
        "r15": 15,
    }


@irdl_attr_definition
class AVXRegisterType(X86RegisterType):
    """
    An x86 register type for AVX512 instructions.
    """

    name = "x86.avxreg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86AVX"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return AVXRegisterType.X86AVX_INDEX_BY_NAME

    X86AVX_INDEX_BY_NAME = {
        "zmm0": 0,
        "zmm1": 1,
        "zmm2": 2,
        "zmm3": 3,
        "zmm4": 4,
        "zmm5": 5,
        "zmm6": 6,
        "zmm7": 7,
        "zmm8": 8,
        "zmm9": 9,
        "zmm10": 10,
        "zmm11": 11,
        "zmm12": 12,
        "zmm13": 13,
        "zmm14": 14,
        "zmm15": 15,
        "zmm16": 16,
        "zmm17": 17,
        "zmm18": 18,
        "zmm19": 19,
        "zmm20": 20,
        "zmm21": 21,
        "zmm22": 22,
        "zmm23": 23,
        "zmm24": 24,
        "zmm25": 25,
        "zmm26": 26,
        "zmm27": 27,
        "zmm28": 28,
        "zmm29": 29,
        "zmm30": 30,
        "zmm31": 31,
    }


R1InvT = TypeVar("R1InvT", bound=X86RegisterType)
R2InvT = TypeVar("R2InvT", bound=X86RegisterType)
R3InvT = TypeVar("R3InvT", bound=X86RegisterType)
# RDInvT = TypeVar("RDInvT", bound=X86RegisterType)
# RSInvT = TypeVar("RSInvT", bound=X86RegisterType)
# RS1InvT = TypeVar("RS1InvT", bound=X86RegisterType)
# RS2InvT = TypeVar("RS2InvT", bound=X86RegisterType)


class Registers(ABC):
    """Namespace for named register constants."""

    RAX = GeneralRegisterType("rax")
    RCX = GeneralRegisterType("rcx")
    RDX = GeneralRegisterType("rdx")
    RBX = GeneralRegisterType("rbx")
    RSP = GeneralRegisterType("rsp")
    RBP = GeneralRegisterType("rbp")
    RSI = GeneralRegisterType("rsi")
    RDI = GeneralRegisterType("rdi")
    R8 = GeneralRegisterType("r8")
    R9 = GeneralRegisterType("r9")
    R10 = GeneralRegisterType("r10")
    R11 = GeneralRegisterType("r11")
    R12 = GeneralRegisterType("r12")
    R13 = GeneralRegisterType("r13")
    R14 = GeneralRegisterType("r14")
    R15 = GeneralRegisterType("r15")

    ZMM0 = AVXRegisterType("zmm0")
    ZMM1 = AVXRegisterType("zmm1")
    ZMM2 = AVXRegisterType("zmm2")
    ZMM3 = AVXRegisterType("zmm3")
    ZMM4 = AVXRegisterType("zmm4")
    ZMM5 = AVXRegisterType("zmm5")
    ZMM6 = AVXRegisterType("zmm6")
    ZMM7 = AVXRegisterType("zmm7")
    ZMM8 = AVXRegisterType("zmm8")
    ZMM9 = AVXRegisterType("zmm9")
    ZMM10 = AVXRegisterType("zmm10")
    ZMM11 = AVXRegisterType("zmm11")
    ZMM12 = AVXRegisterType("zmm12")
    ZMM13 = AVXRegisterType("zmm13")
    ZMM14 = AVXRegisterType("zmm14")
    ZMM15 = AVXRegisterType("zmm15")
    ZMM16 = AVXRegisterType("zmm16")
    ZMM17 = AVXRegisterType("zmm17")
    ZMM18 = AVXRegisterType("zmm18")
    ZMM19 = AVXRegisterType("zmm19")
    ZMM20 = AVXRegisterType("zmm20")
    ZMM21 = AVXRegisterType("zmm21")
    ZMM22 = AVXRegisterType("zmm22")
    ZMM23 = AVXRegisterType("zmm23")
    ZMM24 = AVXRegisterType("zmm24")
    ZMM25 = AVXRegisterType("zmm25")
    ZMM26 = AVXRegisterType("zmm26")
    ZMM27 = AVXRegisterType("zmm27")
    ZMM28 = AVXRegisterType("zmm28")
    ZMM29 = AVXRegisterType("zmm29")
    ZMM30 = AVXRegisterType("zmm30")
    ZMM31 = AVXRegisterType("zmm31")


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "x86.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)


class X86Op(Operation, ABC):
    """
    Base class for operations that can be a part of x86 assembly printing.
    """

    @abstractmethod
    def assembly_line(self) -> str | None:
        raise NotImplementedError()

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        args = cls.parse_unresolved_operands(parser)
        custom_attributes = cls.custom_parse_attributes(parser)
        remaining_attributes = parser.parse_optional_attr_dict()
        # TODO ensure distinct keys for attributes
        attributes = custom_attributes | remaining_attributes
        regions = parser.parse_region_list()
        pos = parser.pos
        operand_types, result_types = cls.parse_op_type(parser)
        operands = parser.resolve_operands(args, operand_types, pos)
        return cls.create(
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
        )

    @classmethod
    def parse_unresolved_operands(cls, parser: Parser) -> list[UnresolvedOperand]:
        """
        Parse a list of comma separated unresolved operands.

        Notice that this method will consume trailing comma.
        """
        if operand := parser.parse_optional_unresolved_operand():
            operands = [operand]
            while parser.parse_optional_punctuation(",") and (
                operand := parser.parse_optional_unresolved_operand()
            ):
                operands.append(operand)
            return operands
        return []

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        """
        Parse attributes with custom syntax. Subclasses may override this method.
        """
        return parser.parse_optional_attr_dict()

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        return func_type.inputs.data, func_type.outputs.data

    def print(self, printer: Printer) -> None:
        if self.operands:
            printer.print(" ")
            printer.print_list(self.operands, printer.print_operand)
        printed_attributes = self.custom_print_attributes(printer)
        unprinted_attributes = {
            name: attr
            for name, attr in self.attributes.items()
            if name not in printed_attributes
        }
        printer.print_op_attributes(unprinted_attributes)
        printer.print_regions(self.regions)
        self.print_op_type(printer)

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        """
        Print attributes with custom syntax. Return the names of the attributes printed. Subclasses may override this method.
        """
        printer.print_op_attributes(self.attributes)
        return self.attributes.keys()

    def print_op_type(self, printer: Printer) -> None:
        printer.print(" : ")
        printer.print_operation_type(self)


AssemblyInstructionArg: TypeAlias = (
    AnyIntegerAttr | LabelAttr | SSAValue | GeneralRegisterType | str | int
)


class X86Instruction(X86Op):
    """
    Base class for operations that can be a part of x86 assembly printing. Must
    represent an instruction in the x86 instruction set.

    The name of the operation will be used as the x86 assembly instruction name.
    """

    comment: StringAttr | None = opt_attr_def(StringAttr)
    """
    An optional comment that will be printed along with the instruction.
    """

    @abstractmethod
    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        """
        The arguments to the instruction, in the order they should be printed in the
        assembly.
        """
        raise NotImplementedError()

    def assembly_instruction_name(self) -> str:
        """
        By default, the name of the instruction is the same as the name of the operation.
        """

        return self.name.split(".", 1)[-1]

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            _assembly_arg_str(arg)
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return _assembly_line(instruction_name, arg_str, self.comment)


class SingleOperandInstruction(IRDLOperation, X86Instruction, ABC):
    """
    Base class for instructions that take a single operand.
    """


class DoubleOperandInstruction(IRDLOperation, X86Instruction, ABC):
    """
    Base class for instructions that take two operands.
    """


class TripleOperandInstruction(IRDLOperation, X86Instruction, ABC):
    """
    Base class for instructions that take three operands.
    """


class RROperation(Generic[R1InvT, R2InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have two registers.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2],
            attributes={
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2


class ROperation(Generic[R1InvT], SingleOperandInstruction):
    """
    A base class for x86 operations that have one register.
    """

    source = opt_operand_def(R1InvT)
    destination = opt_result_def(R1InvT)

    def __init__(
        self,
        source: Operation | SSAValue | None = None,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source],
            attributes={
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.destination,) if self.destination else (self.source,)


# class RdRsOperation(Generic[RDInvT, RS1InvT], IRDLOperation, X86Instruction, ABC):
#     """
#     A base class for x86 operations which has a source register and a destination register.
#     """

#     rd: OpResult = result_def(RDInvT)
#     rs: Operand = operand_def(RS1InvT)

#     def __init__(
#         self,
#         rs: Operation | SSAValue,
#         *,
#         rd: RDInvT,
#         comment: str | StringAttr | None = None,
#     ):
#         if isinstance(comment, str):
#             comment = StringAttr(comment)

#         super().__init__(
#             operands=[rs],
#             attributes={
#                 "comment": comment,
#             },
#             result_types=[rd],
#         )

#     def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
#         return self.rd, self.rs


# class RsOperation(Generic[RS1InvT], IRDLOperation, X86Instruction, ABC):
#     """
#     A base class for x86 operations that have one source register.
#     """

#     rs: Operand = operand_def(RS1InvT)

#     def __init__(
#         self,
#         rs: Operation | SSAValue,
#         *,
#         comment: str | StringAttr | None = None,
#     ):
#         if isinstance(comment, str):
#             comment = StringAttr(comment)

#         super().__init__(
#             operands=[rs],
#             attributes={
#                 "comment": comment,
#             },
#         )

#     def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
#         return (self.rs,)


# class RdOperation(Generic[RDInvT], IRDLOperation, X86Instruction, ABC):
#     """
#     A base class for x86 operations that have one destination register.
#     """

#     rd: OpResult = result_def(RDInvT)

#     def __init__(
#         self,
#         *,
#         rd: RDInvT,
#         comment: str | StringAttr | None = None,
#     ):
#         if isinstance(comment, str):
#             comment = StringAttr(comment)

#         super().__init__(
#             attributes={
#                 "comment": comment,
#             },
#             result_types=[rd],
#         )

#     def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
#         return (self.rd,)


@irdl_op_definition
class AddOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the registers r1 and r2 and stores the result in r1.

    x[r1] = x[r1] + x[r2]
    """

    name = "x86.add"


@irdl_op_definition
class SubOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    subtracts r2 from r1 and stores the result in r1.

    x[r1] = x[r1] - x[r2]
    """

    name = "x86.sub"


@irdl_op_definition
class ImulOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the registers r1 and r2 and stores the result in r1.

    x[r1] = x[r1] * x[r2]
    """

    name = "x86.imul"


@irdl_op_definition
class IdivOp(ROperation[GeneralRegisterType]):
    """
    Divide rdx:rax by x[r1]. Store quotient in rax and store remainder in rdx.
    """

    name = "x86.idiv"


@irdl_op_definition
class NotOp(ROperation[GeneralRegisterType]):
    """
    bitwise not of r1, stored in r1

    x[r1] = ~x[r1]
    """

    name = "x86.not"


@irdl_op_definition
class AndOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r1 and r2, stored in r1

    x[r1] = x[r1] & x[r2]
    """

    name = "x86.and"


@irdl_op_definition
class OrOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r1 and r2, stored in r1

    x[r1] = x[r1] | x[r2]
    """

    name = "x86.or"


@irdl_op_definition
class XorOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r1 and r2, stored in r1

    x[r1] = x[r1] ^ x[r2]
    """

    name = "x86.xor"


@irdl_op_definition
class MovOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value of r1 into r2.

    x[r1] = x[r2]
    """

    name = "x86.mov"


@irdl_op_definition
class PushOp(ROperation[GeneralRegisterType]):
    """
    Decreases %rsp and places r1 at the new memory location pointed to by %rsp.
    """

    name = "x86.push"


@irdl_op_definition
class PopOp(ROperation[GeneralRegisterType]):
    """
    Copies the value at the top of the stack into r1 and increases %rsp.
    """

    name = "x86.pop"


class RRROperation(Generic[R1InvT, R2InvT, R3InvT], TripleOperandInstruction):
    """
    A base class for x86 operations that have three registers.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    r3 = operand_def(R3InvT)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        r3: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2, r3],
            attributes={
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2, self.r3


class RROffOperation(Generic[R1InvT, R2InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have two registers and an offset.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    offset: AnyIntegerAttr = attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)  # I have no clue why that is 12
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2


@irdl_op_definition
class DirectiveOp(IRDLOperation, X86Op):
    """
    The directive operation is used to represent a directive in the assembly code. (e.g. .globl; .type etc)
    """

    name = "x86.directive"

    directive: StringAttr = attr_def(StringAttr)
    value: StringAttr | None = opt_attr_def(StringAttr)

    def __init__(
        self,
        directive: str | StringAttr,
        value: str | StringAttr | None,
    ):
        if isinstance(directive, str):
            directive = StringAttr(directive)
        if isinstance(value, str):
            value = StringAttr(value)

        super().__init__(
            attributes={
                "directive": directive,
                "value": value,
            },
        )

    def assembly_line(self) -> str | None:
        if self.value is not None and self.value.data:
            arg_str = _assembly_arg_str(self.value.data)
        else:
            arg_str = ""

        return _assembly_line(self.directive.data, arg_str, is_indented=False)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["directive"] = StringAttr(
            parser.parse_str_literal("Expected directive")
        )
        if (value := parser.parse_optional_str_literal()) is not None:
            attributes["value"] = StringAttr(value)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(" ")
        printer.print_string_literal(self.directive.data)
        if self.value is not None:
            printer.print(" ")
            printer.print_string_literal(self.value.data)
        return {"directive", "value"}

    def print_op_type(self, printer: Printer) -> None:
        return

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        return (), ()


@irdl_op_definition
class Vfmadd231pdOp(RRROperation[AVXRegisterType, AVXRegisterType, AVXRegisterType]):
    """
    Multiply packed double-precision floating-point elements in r2 and r3, add the intermediate result to r1, and store the final result in r1.
    """

    name = "x86.vfmadd231pd"


@irdl_op_definition
class VmovapdOp(RROffOperation[AVXRegisterType, GeneralRegisterType]):
    """
    Move aligned packed double-precision floating-point elements.
    """

    name = "x86.vmovapd"

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        offset = _assembly_arg_str(self.offset)
        return _assembly_line(
            instruction_name, f"{destination}, {offset}({source})", self.comment
        )


@irdl_op_definition
class VbroadcastsdOp(RROffOperation[AVXRegisterType, GeneralRegisterType]):
    """
    Broadcast scalar double-precision floating-point element.
    """

    name = "x86.vbroadcastsd"

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        offset = _assembly_arg_str(self.offset)
        return _assembly_line(
            instruction_name, f"{destination}, {offset}({source})", self.comment
        )


# region Assembly printing


def _append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


def _assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isa(arg, AnyIntegerAttr):
        return f"{arg.value.data}"
    elif isinstance(arg, int):
        return f"{arg}"
    elif isinstance(arg, LabelAttr):
        return arg.data
    elif isinstance(arg, str):
        return arg
    elif isinstance(arg, GeneralRegisterType):
        return arg.register_name
    elif isinstance(arg, AVXRegisterType):
        return arg.register_name
    else:
        if isinstance(arg.type, GeneralRegisterType):
            reg = arg.type.register_name
            return reg
        elif isinstance(arg.type, AVXRegisterType):
            reg = arg.type.register_name
            return reg
        else:
            assert False, f"{arg.type}"


def _assembly_line(
    name: str,
    arg_str: str,
    comment: StringAttr | None = None,
    is_indented: bool = True,
) -> str:
    code = "    " if is_indented else ""
    code += name
    if arg_str:
        code += f" {arg_str}"
    code = _append_comment(code, comment)
    return code


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        assert isinstance(op, X86Op), f"{op}"
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)


def x86_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()


class GetAnyRegisterOperation(Generic[R1InvT], IRDLOperation, X86Op):
    """
    This instruction allows us to create an SSAValue with for a given register name. This
    is useful for bridging the x86 convention that stores the result of function calls
    in `eax` into SSA form.

    """

    result = result_def(R1InvT)

    def __init__(
        self,
        register_type: R1InvT,
    ):
        super().__init__(result_types=[register_type])

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[GeneralRegisterType]):
    name = "x86.get_register"


@irdl_op_definition
class GetAVXRegisterOp(GetAnyRegisterOperation[AVXRegisterType]):
    name = "x86.get_avx_register"


X86 = Dialect(
    "x86",
    [
        AddOp,
        SubOp,
        ImulOp,
        IdivOp,
        NotOp,
        AndOp,
        OrOp,
        XorOp,
        MovOp,
        PushOp,
        PopOp,
        Vfmadd231pdOp,
        VmovapdOp,
        VbroadcastsdOp,
        DirectiveOp,
        GetRegisterOp,
        GetAVXRegisterOp,
    ],
    [
        GeneralRegisterType,
        AVXRegisterType,
        LabelAttr,
    ],
)
# endregion
