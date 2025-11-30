from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from io import StringIO
from typing import IO, Annotated, Generic, Literal, TypeAlias

from typing_extensions import Self, TypeVar

from xdsl.backend.assembly_printer import (
    AssemblyPrintable,
    AssemblyPrinter,
    OneLineAssemblyPrintable,
)
from xdsl.backend.register_allocatable import (
    HasRegisterConstraints,
    RegisterConstraints,
)
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect, RegisterType
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    NoneAttr,
    Signedness,
    StringAttr,
    UnitAttr,
    i32,
)
from xdsl.dialects.utils import FastMathAttrBase, FastMathFlag
from xdsl.interfaces import ConstantLikeInterface
from xdsl.ir import (
    Attribute,
    Block,
    Data,
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    IsolatedFromAbove,
    IsTerminator,
    NoTerminator,
    Pure,
)
from xdsl.utils.exceptions import VerifyException


def is_non_zero(reg: IntRegisterType) -> bool:
    """
    Returns True if the register is allocated, and is not the x0/ZERO register.
    """
    return (
        reg.is_allocated and not isinstance(reg.index, NoneAttr) and reg.index.data != 0
    )


@irdl_attr_definition
class FastMathFlagsAttr(FastMathAttrBase):
    """
    riscv.fastmath is a mirror of LLVMs fastmath flags.
    """

    name = "riscv.fastmath"

    def __init__(self, flags: None | Sequence[FastMathFlag] | Literal["none", "fast"]):
        # irdl_attr_definition defines an __init__ if none is defined, so we need to
        # explicitely define one here.
        super().__init__(flags)


class RISCVRegisterType(RegisterType):
    """
    A RISC-V register type.
    """

    @classmethod
    @abstractmethod
    def a_register(cls, index: int) -> Self:
        raise NotImplementedError()


_RV32I_ABI_INDEX_BY_NAME = {
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
    "t6": 31,
}
_RV32I_X_INDEX_BY_NAME = {f"x{i}": i for i in range(32)}
RV32I_INDEX_BY_NAME = _RV32I_X_INDEX_BY_NAME | _RV32I_ABI_INDEX_BY_NAME


@irdl_attr_definition
class IntRegisterType(RISCVRegisterType):
    """
    A RISC-V register type.
    """

    name = "riscv.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return RV32I_INDEX_BY_NAME

    @classmethod
    def a_register(cls, index: int) -> IntRegisterType:
        return Registers.A[index]

    @classmethod
    def infinite_register_prefix(cls):
        return "j_"

    # This class variable is created and exclusively accessed in `abi_name_by_index`.
    # _ALLOCATABLE_REGISTERS: ClassVar[tuple[IntRegisterType, ...]]

    @classmethod
    def allocatable_registers(cls):
        if not hasattr(cls, "_ALLOCATABLE_REGISTERS"):
            cls._ALLOCATABLE_REGISTERS = (*Registers.T, *Registers.A)
        return cls._ALLOCATABLE_REGISTERS


_RV32F_ABI_INDEX_BY_NAME = {
    "ft0": 0,
    "ft1": 1,
    "ft2": 2,
    "ft3": 3,
    "ft4": 4,
    "ft5": 5,
    "ft6": 6,
    "ft7": 7,
    "fs0": 8,
    "fs1": 9,
    "fa0": 10,
    "fa1": 11,
    "fa2": 12,
    "fa3": 13,
    "fa4": 14,
    "fa5": 15,
    "fa6": 16,
    "fa7": 17,
    "fs2": 18,
    "fs3": 19,
    "fs4": 20,
    "fs5": 21,
    "fs6": 22,
    "fs7": 23,
    "fs8": 24,
    "fs9": 25,
    "fs10": 26,
    "fs11": 27,
    "ft8": 28,
    "ft9": 29,
    "ft10": 30,
    "ft11": 31,
}
_RV32F_F_INDEX_BY_NAME = {f"f{i}": i for i in range(32)}
RV32F_INDEX_BY_NAME = _RV32F_F_INDEX_BY_NAME | _RV32F_ABI_INDEX_BY_NAME


@irdl_attr_definition
class FloatRegisterType(RISCVRegisterType):
    """
    A RISC-V register type.
    """

    name = "riscv.freg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return RV32F_INDEX_BY_NAME

    @classmethod
    def a_register(cls, index: int) -> FloatRegisterType:
        return Registers.FA[index]

    @classmethod
    def infinite_register_prefix(cls):
        return "fj_"

    # This class variable is created and exclusively accessed in `abi_name_by_index`.
    # _ALLOCATABLE_REGISTERS: ClassVar[tuple[FloatRegisterType, ...]]

    @classmethod
    def allocatable_registers(cls):
        if not hasattr(cls, "_ALLOCATABLE_REGISTERS"):
            cls._ALLOCATABLE_REGISTERS = (*Registers.FT, *Registers.FA)
        return cls._ALLOCATABLE_REGISTERS


RDInvT = TypeVar("RDInvT", bound=RISCVRegisterType)
RSInvT = TypeVar("RSInvT", bound=RISCVRegisterType)
RS1InvT = TypeVar("RS1InvT", bound=RISCVRegisterType)
RS2InvT = TypeVar("RS2InvT", bound=RISCVRegisterType)


class Registers(ABC):
    """Namespace for named register constants."""

    UNALLOCATED_INT = IntRegisterType.unallocated()
    ZERO = IntRegisterType.from_name("zero")
    RA = IntRegisterType.from_name("ra")
    SP = IntRegisterType.from_name("sp")
    GP = IntRegisterType.from_name("gp")
    TP = IntRegisterType.from_name("tp")
    T0 = IntRegisterType.from_name("t0")
    T1 = IntRegisterType.from_name("t1")
    T2 = IntRegisterType.from_name("t2")
    FP = IntRegisterType.from_name("fp")
    S0 = IntRegisterType.from_name("s0")
    S1 = IntRegisterType.from_name("s1")
    A0 = IntRegisterType.from_name("a0")
    A1 = IntRegisterType.from_name("a1")
    A2 = IntRegisterType.from_name("a2")
    A3 = IntRegisterType.from_name("a3")
    A4 = IntRegisterType.from_name("a4")
    A5 = IntRegisterType.from_name("a5")
    A6 = IntRegisterType.from_name("a6")
    A7 = IntRegisterType.from_name("a7")
    S2 = IntRegisterType.from_name("s2")
    S3 = IntRegisterType.from_name("s3")
    S4 = IntRegisterType.from_name("s4")
    S5 = IntRegisterType.from_name("s5")
    S6 = IntRegisterType.from_name("s6")
    S7 = IntRegisterType.from_name("s7")
    S8 = IntRegisterType.from_name("s8")
    S9 = IntRegisterType.from_name("s9")
    S10 = IntRegisterType.from_name("s10")
    S11 = IntRegisterType.from_name("s11")
    T3 = IntRegisterType.from_name("t3")
    T4 = IntRegisterType.from_name("t4")
    T5 = IntRegisterType.from_name("t5")
    T6 = IntRegisterType.from_name("t6")

    UNALLOCATED_FLOAT = FloatRegisterType.unallocated()
    FT0 = FloatRegisterType.from_name("ft0")
    FT1 = FloatRegisterType.from_name("ft1")
    FT2 = FloatRegisterType.from_name("ft2")
    FT3 = FloatRegisterType.from_name("ft3")
    FT4 = FloatRegisterType.from_name("ft4")
    FT5 = FloatRegisterType.from_name("ft5")
    FT6 = FloatRegisterType.from_name("ft6")
    FT7 = FloatRegisterType.from_name("ft7")
    FS0 = FloatRegisterType.from_name("fs0")
    FS1 = FloatRegisterType.from_name("fs1")
    FA0 = FloatRegisterType.from_name("fa0")
    FA1 = FloatRegisterType.from_name("fa1")
    FA2 = FloatRegisterType.from_name("fa2")
    FA3 = FloatRegisterType.from_name("fa3")
    FA4 = FloatRegisterType.from_name("fa4")
    FA5 = FloatRegisterType.from_name("fa5")
    FA6 = FloatRegisterType.from_name("fa6")
    FA7 = FloatRegisterType.from_name("fa7")
    FS2 = FloatRegisterType.from_name("fs2")
    FS3 = FloatRegisterType.from_name("fs3")
    FS4 = FloatRegisterType.from_name("fs4")
    FS5 = FloatRegisterType.from_name("fs5")
    FS6 = FloatRegisterType.from_name("fs6")
    FS7 = FloatRegisterType.from_name("fs7")
    FS8 = FloatRegisterType.from_name("fs8")
    FS9 = FloatRegisterType.from_name("fs9")
    FS10 = FloatRegisterType.from_name("fs10")
    FS11 = FloatRegisterType.from_name("fs11")
    FT8 = FloatRegisterType.from_name("ft8")
    FT9 = FloatRegisterType.from_name("ft9")
    FT10 = FloatRegisterType.from_name("ft10")
    FT11 = FloatRegisterType.from_name("ft11")

    # register classes:

    A = (A0, A1, A2, A3, A4, A5, A6, A7)
    T = (T0, T1, T2, T3, T4, T5, T6)
    S = (S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11)

    FA = (FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7)
    FT = (FT0, FT1, FT2, FT3, FT4, FT5, FT6, FT7, FT8, FT9, FT10, FT11)
    FS = (FS0, FS1, FS2, FS3, FS4, FS5, FS6, FS7, FS8, FS9, FS10, FS11)


ui5 = IntegerType(5, Signedness.UNSIGNED)
si20 = IntegerType(20, Signedness.SIGNED)
si12 = IntegerType(12, Signedness.SIGNED)
i12 = IntegerType(12, Signedness.SIGNLESS)
i20 = IntegerType(20, Signedness.SIGNLESS)
UImm5Attr = IntegerAttr[Annotated[IntegerType, ui5]]
SImm12Attr = IntegerAttr[Annotated[IntegerType, si12]]
SImm20Attr = IntegerAttr[Annotated[IntegerType, si20]]
Imm12Attr = IntegerAttr[Annotated[IntegerType, i12]]
Imm20Attr = IntegerAttr[Annotated[IntegerType, i20]]
Imm32Attr = IntegerAttr[Annotated[IntegerType, i32]]


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "riscv.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)


class RISCVAsmOperation(
    HasRegisterConstraints, IRDLOperation, OneLineAssemblyPrintable, ABC
):
    """
    Base class for operations that can be a part of RISC-V assembly printing.
    """

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(self.operands, self.results, ())


class RISCVCustomFormatOperation(IRDLOperation, ABC):
    """
    Base class for RISC-V operations that specialize their custom format.
    """

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
            printer.print_string(" ")
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

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        """
        Print attributes with custom syntax. Return the names of the attributes printed. Subclasses may override this method.
        """
        printer.print_op_attributes(self.attributes)
        return self.attributes.keys()

    def print_op_type(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_operation_type(self)


AssemblyInstructionArg: TypeAlias = (
    IntegerAttr | LabelAttr | SSAValue | RegisterType | str
)


class RISCVInstruction(RISCVAsmOperation, ABC):
    """
    Base class for operations that can be a part of RISC-V assembly printing. Must
    represent an instruction in the RISC-V instruction set, and have the following format:

    name arg0, arg1, arg2           # comment

    The name of the operation will be used as the RISC-V assembly instruction name.
    """

    comment = opt_attr_def(StringAttr)
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

        return Dialect.split_name(self.name)[1]

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            _assembly_arg_str(arg)
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return AssemblyPrinter.assembly_line(instruction_name, arg_str, self.comment)


# region Assembly printing


def _assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, SSAValue):
        if not isinstance(t := arg.type, RegisterType):
            raise ValueError(f"Unexpected register type {t}")
        return t.register_name.data
    elif isinstance(arg, IntegerAttr):
        return f"{arg.value.data}"
    elif isinstance(arg, LabelAttr):
        return arg.data
    elif isinstance(arg, RegisterType):
        return arg.register_name.data

    return arg


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    printer = AssemblyPrinter(stream=output)
    printer.print_module(module)


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()


# endregion

# region Base Operation classes


class RdRsRsOperation(
    RISCVCustomFormatOperation, RISCVInstruction, ABC, Generic[RDInvT, RS1InvT, RS2InvT]
):
    """
    A base class for RISC-V operations that have one destination register, and two source
    registers.

    This is called R-Type in the RISC-V specification.
    """

    rd = result_def(RDInvT)
    rs1 = operand_def(RS1InvT)
    rs2 = operand_def(RS2InvT)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: RDInvT = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.rs2


class RdRsRsIntegerOperation(
    RdRsRsOperation[IntRegisterType, RS1InvT, RS2InvT], ABC, Generic[RS1InvT, RS2InvT]
):
    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(rs1, rs2, rd=rd, comment=comment)


class RdRsRsFloatOperation(
    RdRsRsOperation[FloatRegisterType, RS1InvT, RS2InvT], ABC, Generic[RS1InvT, RS2InvT]
):
    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: FloatRegisterType = Registers.UNALLOCATED_FLOAT,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(rs1, rs2, rd=rd, comment=comment)


class RdRsRsFloatOperationWithFastMath(
    RISCVCustomFormatOperation, RISCVInstruction, ABC
):
    """
    A base class for RISC-V operations that have one destination floating-point register,
    and two source floating-point registers and can be annotated with fastmath flags.

    This is called R-Type in the RISC-V specification.
    """

    rd = result_def(FloatRegisterType)
    rs1 = operand_def(FloatRegisterType)
    rs2 = operand_def(FloatRegisterType)
    fastmath = opt_attr_def(FastMathFlagsAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: FloatRegisterType = Registers.UNALLOCATED_FLOAT,
        fastmath: FastMathFlagsAttr | None = None,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "fastmath": fastmath,
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.rs2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        flags = FastMathFlagsAttr("none")
        if parser.parse_optional_keyword("fastmath") is not None:
            flags = FastMathFlagsAttr(FastMathFlagsAttr.parse_parameter(parser))
        attributes["fastmath"] = flags
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        if self.fastmath is not None and self.fastmath != FastMathFlagsAttr("none"):
            printer.print_string(" fastmath")
            self.fastmath.print_parameter(printer)
        return {"fastmath"}


class RdImmIntegerOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one destination register, and one
    immediate operand (e.g. U-Type and J-Type instructions in the RISC-V spec).
    """

    rd = result_def(IntRegisterType)
    immediate = attr_def(base(Imm20Attr) | base(LabelAttr))

    def __init__(
        self,
        immediate: int | IntegerAttr | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i20)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            result_types=[rd],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i20)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class RdImmJumpOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    In the RISC-V spec, this is the same as `RdImmOperation`. For jumps, the `rd` register
    is neither an operand, because the stored value is overwritten, nor a result value,
    because the value in `rd` is not defined after the jump back. So the `rd` makes the
    most sense as an attribute.
    """

    rd = opt_attr_def(IntRegisterType)
    """
    The rd register here is not a register storing the result, rather the register where
    the program counter is stored before jumping.
    """
    immediate = attr_def(base(SImm20Attr) | base(LabelAttr))

    def __init__(
        self,
        immediate: int | SImm20Attr | str | LabelAttr,
        *,
        rd: IntRegisterType | None = None,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, si20)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            attributes={
                "immediate": immediate,
                "rd": rd,
                "comment": comment,
            }
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.rd, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, si20)
        if parser.parse_optional_punctuation(","):
            attributes["rd"] = parser.parse_attribute()
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        print_immediate_value(printer, self.immediate)
        if self.rd is not None:
            printer.print_string(", ")
            printer.print_attribute(self.rd)
        return {"immediate", "rd"}

    def print_op_type(self, printer: Printer) -> None:
        return

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        return (), ()


class RdRsImmIntegerOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one destination register, one source
    register and one immediate operand.

    This is called I-Type in the RISC-V specification.
    """

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(base(SImm12Attr) | base(LabelAttr))

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | SImm12Attr | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, si12)
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
        attributes["immediate"] = parse_immediate_value(parser, si12)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class RdRsImmShiftOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one destination register, one source
    register and one immediate operand.

    This is called I-Type in the RISC-V specification.

    Shifts by a constant are encoded as a specialization of the I-type format.
    The shift amount is encoded in the lower 5 bits of the I-immediate field for RV32

    For RV32I, SLLI, SRLI, and SRAI generate an illegal instruction exception if
    imm[5] 6 != 0 but the shift amount is encoded in the lower 6 bits of the I-immediate field for RV64I.
    """

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


class RdRsImmJumpOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one destination register, one source
    register and one immediate operand.

    This is called I-Type in the RISC-V specification.

    In the RISC-V spec, this is the same as `RdRsImmOperation`. For jumps, the `rd` register
    is neither an operand, because the stored value is overwritten, nor a result value,
    because the value in `rd` is not defined after the jump back. So the `rd` makes the
    most sense as an attribute.
    """

    rs1 = operand_def(IntRegisterType)
    rd = opt_attr_def(IntRegisterType)
    """
    The rd register here is not a register storing the result, rather the register where
    the program counter is stored before jumping.
    """
    immediate = attr_def(base(SImm12Attr) | base(LabelAttr))

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | SImm12Attr | str | LabelAttr,
        *,
        rd: IntRegisterType | None = None,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, si12)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)

        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1],
            attributes={
                "immediate": immediate,
                "rd": rd,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.rd, self.rs1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, si12)
        if parser.parse_optional_punctuation(","):
            attributes["rd"] = parser.parse_attribute()
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        if self.rd is not None:
            printer.print_string(", ")
            printer.print_attribute(self.rd)
        return {"immediate", "rd"}


class RdRsOperation(
    RISCVCustomFormatOperation, RISCVInstruction, ABC, Generic[RDInvT, RSInvT]
):
    """
    A base class for RISC-V pseudo-instructions that have one destination register and one
    source register.
    """

    rd = result_def(RDInvT)
    rs = operand_def(RSInvT)

    def __init__(
        self,
        rs: Operation | SSAValue,
        *,
        rd: RDInvT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs],
            result_types=[rd],
            attributes={"comment": comment},
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs


class RdRsIntegerOperation(
    RdRsOperation[IntRegisterType, RSInvT], ABC, Generic[RSInvT]
):
    def __init__(
        self,
        rs: Operation | SSAValue,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(rs, rd=rd, comment=comment)


class RdRsFloatOperation(
    RdRsOperation[FloatRegisterType, RSInvT], ABC, Generic[RSInvT]
):
    def __init__(
        self,
        rs: Operation | SSAValue,
        *,
        rd: FloatRegisterType = Registers.UNALLOCATED_FLOAT,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(rs, rd=rd, comment=comment)


class RsRsOffIntegerOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one source register and a destination
    register, and an offset.

    This is called B-Type in the RISC-V specification.
    """

    rs1 = operand_def(IntRegisterType)
    rs2 = operand_def(IntRegisterType)
    offset = attr_def(base(SImm12Attr) | base(LabelAttr))

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        offset: int | SImm12Attr | LabelAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, si12)
        if isinstance(offset, str):
            offset = LabelAttr(offset)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "offset": offset,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.offset

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = parse_immediate_value(parser, si12)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.offset)
        return {"offset"}


class RsRsImmIntegerOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have two source registers and an
    immediate.

    This is called S-Type in the RISC-V specification.
    """

    rs1 = operand_def(IntRegisterType)
    rs2 = operand_def(IntRegisterType)
    immediate = attr_def(SImm12Attr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: int | Imm12Attr | str | LabelAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, si12)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, si12)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class RsRsIntegerOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have two source
    registers.
    """

    rs1 = operand_def(IntRegisterType)
    rs2 = operand_def(IntRegisterType)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2


class NullaryOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have neither sources nor destinations.
    """

    def __init__(
        self,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            attributes={
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return ()

    @classmethod
    def parse_unresolved_operands(cls, parser: Parser) -> list[UnresolvedOperand]:
        return []

    def print_op_type(self, printer: Printer) -> None:
        return

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        return (), ()


class CsrReadWriteOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations performing a swap to/from a CSR.

    The 'writeonly' attribute controls the actual behaviour of the operation:
    * when True, the operation writes the rs value to the CSR but never reads it and
      in this case rd *must* be allocated to x0
    * when False, a proper atomic swap is performed and the previous CSR value is
      returned in rd
    """

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    csr = attr_def(IntegerAttr)
    writeonly = opt_attr_def(UnitAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        csr: IntegerAttr,
        *,
        writeonly: bool = False,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            attributes={
                "csr": csr,
                "writeonly": UnitAttr() if writeonly else None,
                "comment": comment,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if not self.writeonly:
            return
        if is_non_zero(self.rd.type):
            raise VerifyException(
                "When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rd.type.register_name.data}'"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.csr, self.rs1

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["csr"] = IntegerAttr(
            parser.parse_integer(allow_boolean=False, context_msg="Expected csr"),
            IntegerType(32),
        )
        if parser.parse_optional_punctuation(",") is not None:
            if (flag := parser.parse_str_literal("Expected 'w' flag")) != "w":
                parser.raise_error(f"Expected 'w' flag, got '{flag}'")
            attributes["writeonly"] = UnitAttr()
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        self.csr.print_without_type(printer)
        if self.writeonly is not None:
            printer.print_string(', "w"')
        return {"csr", "writeonly"}


class CsrBitwiseOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations performing a masked bitwise operation on the
    CSR while returning the original value.

    The 'readonly' attribute controls the actual behaviour of the operation:
    * when True, the operation is guaranteed to have no side effects that can
      be potentially related to writing to a CSR; in this case rs *must be
      allocated to x0*
    * when False, the bitwise operations is performed and any side effect related
      to writing to a CSR takes place even if the mask in rs has no actual bits set.
    """

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    csr = attr_def(IntegerAttr)
    readonly = opt_attr_def(UnitAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        csr: IntegerAttr,
        *,
        readonly: bool = False,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[rs1],
            attributes={
                "csr": csr,
                "readonly": UnitAttr() if readonly else None,
                "comment": comment,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if not self.readonly:
            return
        assert isinstance(self.rs1.type, IntRegisterType)
        if is_non_zero(self.rs1.type):
            raise VerifyException(
                "When in 'readonly' mode, source must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rs1.type.register_name.data}'"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.csr, self.rs1

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["csr"] = IntegerAttr(
            parser.parse_integer(allow_boolean=False, context_msg="Expected csr"),
            IntegerType(32),
        )
        if parser.parse_optional_punctuation(",") is not None:
            if (flag := parser.parse_str_literal("Expected 'r' flag")) != "r":
                parser.raise_error(f"Expected 'r' flag, got '{flag}'")
            attributes["readonly"] = UnitAttr()
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        self.csr.print_without_type(printer)
        if self.readonly is not None:
            printer.print_string(', "r"')
        return {"csr", "readonly"}


class CsrReadWriteImmOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations performing a write immediate to/read from a CSR.

    The 'writeonly' attribute controls the actual behaviour of the operation:
    * when True, the operation writes the rs value to the CSR but never reads it and
      in this case rd *must* be allocated to x0
    * when False, a proper atomic swap is performed and the previous CSR value is
      returned in rd
    """

    rd = result_def(IntRegisterType)
    csr = attr_def(IntegerAttr)
    immediate = attr_def(IntegerAttr)
    writeonly = opt_attr_def(UnitAttr)

    def __init__(
        self,
        csr: IntegerAttr,
        immediate: IntegerAttr,
        *,
        writeonly: bool = False,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            attributes={
                "csr": csr,
                "immediate": immediate,
                "writeonly": UnitAttr() if writeonly else None,
                "comment": comment,
            },
            result_types=[rd],
        )

    def verify_(self) -> None:
        if self.writeonly is None:
            return
        if is_non_zero(self.rd.type):
            raise VerifyException(
                "When in 'writeonly' mode, destination must be register x0 (a.k.a. 'zero'), "
                f"not '{self.rd.type.register_name.data}'"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.rd, self.csr, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["csr"] = IntegerAttr(
            parser.parse_integer(allow_boolean=False, context_msg="Expected csr"),
            IntegerType(32),
        )
        parser.parse_punctuation(",")
        attributes["immediate"] = parse_immediate_value(parser, IntegerType(32))
        if parser.parse_optional_punctuation(",") is not None:
            if (flag := parser.parse_str_literal("Expected 'w' flag")) != "w":
                parser.raise_error(f"Expected 'w' flag, got '{flag}'")
            attributes["writeonly"] = UnitAttr()
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        self.csr.print_without_type(printer)
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        if self.writeonly is not None:
            printer.print_string(', "w"')
        return {"csr", "immediate", "writeonly"}


class CsrBitwiseImmOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations performing a masked bitwise operation on the
    CSR while returning the original value. The bitmask is specified in the 'immediate'
    attribute.

    The 'immediate' attribute controls the actual behaviour of the operation:
    * when equals to zero, the operation is guaranteed to have no side effects
      that can be potentially related to writing to a CSR;
    * when not equal to zero, any side effect related to writing to a CSR takes
      place.
    """

    rd = result_def(IntRegisterType)
    csr = attr_def(IntegerAttr)
    immediate = attr_def(IntegerAttr)

    def __init__(
        self,
        csr: IntegerAttr,
        immediate: IntegerAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            attributes={
                "csr": csr,
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.csr, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["csr"] = IntegerAttr(
            parser.parse_integer(allow_boolean=False, context_msg="Expected csr"),
            IntegerType(32),
        )
        parser.parse_punctuation(",")
        attributes["immediate"] = parse_immediate_value(parser, IntegerType(32))
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        self.csr.print_without_type(printer)
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"csr", "immediate"}


# endregion

# region RV32I/RV64I: 2.4 Integer Computational Instructions

## Integer Register-Immediate Instructions


class AddiOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            AddImmediateConstant,
            AddImmediateZero,
        )

        return (
            AddImmediateZero(),
            AddImmediateConstant(),
        )


@irdl_op_definition
class AddiOp(RdRsImmIntegerOperation):
    """
    Adds the sign-extended 12-bit immediate to register rs1.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] + sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#addi).
    """

    name = "riscv.addi"

    traits = traits_def(Pure(), AddiOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SltiOp(RdRsImmIntegerOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the sign-extended
    immediate when both are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slti).
    """

    name = "riscv.slti"


@irdl_op_definition
class SltiuOp(RdRsImmIntegerOperation):
    """
    Place the value 1 in register rd if register rs1 is less than the immediate when
    both are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltiu).
    """

    name = "riscv.sltiu"


class AndiOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            AndiImmediate,
        )

        return (AndiImmediate(),)


@irdl_op_definition
class AndiOp(RdRsImmIntegerOperation):
    """
    Performs bitwise AND on register rs1 and the sign-extended 12-bit
    immediate and place the result in rd.

    x[rd] = x[rs1] & sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#andi).
    """

    name = "riscv.andi"
    traits = traits_def(AndiOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class OriOp(RdRsImmIntegerOperation):
    """
    Performs bitwise OR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] | sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#ori).
    """

    name = "riscv.ori"


@irdl_op_definition
class XoriOp(RdRsImmIntegerOperation):
    """
    Performs bitwise XOR on register rs1 and the sign-extended 12-bit immediate and place
    the result in rd.

    x[rd] = x[rs1] ^ sext(immediate)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xori).
    """

    name = "riscv.xori"


class SlliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ShiftLeftbyZero,
            ShiftLeftImmediate,
        )

        return (ShiftLeftImmediate(), ShiftLeftbyZero())


@irdl_op_definition
class SlliOp(RdRsImmShiftOperation):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] << shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slli).
    """

    name = "riscv.slli"

    traits = traits_def(SlliOpHasCanonicalizationPatternsTrait())


class SrliOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import ShiftRightbyZero

        return (ShiftRightbyZero(),)


@irdl_op_definition
class SrliOp(RdRsImmShiftOperation):
    """
    Performs logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>u shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srli).
    """

    name = "riscv.srli"

    traits = traits_def(SrliOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraiOp(RdRsImmShiftOperation):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of the immediate.

    x[rd] = x[rs1] >>s shamt

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srai).
    """

    name = "riscv.srai"


@irdl_op_definition
class AddiwOp(RdRsImmIntegerOperation):
    """
    Adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd.
    Overflows are ignored and the result is the low 32 bits of the result sign-extended to 64 bits.
    ```
    x[rd] = sext((x[rs1] + sext(immediate))[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64i.html#addiw).
    """

    name = "riscv.addiw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliwOp(RdRsImmShiftOperation):
    """
    Performs logical left shift on the 32-bit of value in register rs1 by the
    shift amount held in the lower 5 bits of the immediate.
    ```
    x[rd] = sext((x[rs1] << shamt)[31:0])
    ```
    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slliw).
    """

    name = "riscv.slliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrliwOp(RdRsImmShiftOperation):
    """
    Performs logical right shift on the 32-bit of value in register rs1 by the shift amount held in the
    lower 5 bits of the immediate.
    ```
    x[rd] = sext(x[rs1][31:0] >>u shamt)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srliw).
    """

    name = "riscv.srliw"

    traits = traits_def(Pure())


@irdl_op_definition
class SraiwOp(RdRsImmIntegerOperation):
    """
    Performs arithmetic right shift on the 32-bit of value in register rs1 by the shift amount held
    in the lower 5 bits of the immediate.
    ```
    x[rd] = sext(x[rs1][31:0] >>s shamt)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sraiw).
    """

    name = "riscv.sraiw"

    traits = traits_def(Pure())


@irdl_op_definition
class AddwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Adds the 32-bit of registers rs1 and 32-bit of register rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the low 32-bits of the result is sign-extended to 64-bits and
    written to the destination register.
    ```
    x[rd] = sext((x[rs1] + x[rs2])[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#addw).
    """

    name = "riscv.addw"

    traits = traits_def(Pure())


@irdl_op_definition
class SubwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Subtract the 32-bit of registers rs1 and 32-bit of register rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the low 32-bits of the result is sign-extended to 64-bits
    and written to the destination register.
    ```
    x[rd] = sext((x[rs1] - x[rs2])[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#subw).
    """

    name = "riscv.subw"

    traits = traits_def(Pure())


@irdl_op_definition
class SllwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs logical left shift on the low 32-bits value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2 and produce 32-bit results and written to the destination register rd.
    ```
    x[rd] = sext((x[rs1] << x[rs2][4:0])[31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sllw).
    """

    name = "riscv.sllw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrlwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs logical right shift on the low 32-bits value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2 and produce 32-bit results and written to the destination
    register rd.
    ```
    x[rd] = sext(x[rs1][31:0] >>u x[rs2][4:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srlw).
    """

    name = "riscv.srlw"

    traits = traits_def(Pure())


@irdl_op_definition
class SrawOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs arithmetic right shift on the low 32-bits value in register rs1 by the shift amount held in the lower
    5 bits of register rs2 and produce 32-bit results and written to the destination register rd.
    ```
    x[rd] = sext(x[rs1][31:0] >>s x[rs2][4:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sraw).
    """

    name = "riscv.sraw"

    traits = traits_def(Pure())


@irdl_op_definition
class LuiOp(RdImmIntegerOperation):
    """
    Build 32-bit constants and uses the U-type format. LUI places the U-immediate value
    in the top 20 bits of the destination register rd, filling in the lowest 12 bits with zeros.

    x[rd] = sext(immediate[31:12] << 12)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lui).
    """

    name = "riscv.lui"


@irdl_op_definition
class AuipcOp(RdImmIntegerOperation):
    """
    Build pc-relative addresses and uses the U-type format. AUIPC forms a 32-bit offset
    from the 20-bit U-immediate, filling in the lowest 12 bits with zeros, adds this
    offset to the pc, then places the result in register rd.

    x[rd] = pc + sext(immediate[31:12] << 12)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#auipc).
    """

    name = "riscv.auipc"


class MVHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            RemoveRedundantMv,
        )

        return (RemoveRedundantMv(),)


@irdl_op_definition
class MVOp(RdRsIntegerOperation[IntRegisterType]):
    """
    A pseudo instruction to copy contents of one int register to another.

    Equivalent to `addi rd, rs, 0`
    """

    name = "riscv.mv"

    traits = traits_def(
        Pure(),
        MVHasCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class SeqzOp(RdRsIntegerOperation[IntRegisterType]):
    """
    A pseudo instruction that sets the destination register to 1 if the source register is equal to zero.

    Equivalent to `sltiu rd, rs, 1
    """

    name = "riscv.seqz"


@irdl_op_definition
class SnezOp(RdRsIntegerOperation[IntRegisterType]):
    """
    A pseudo instruction that sets the destination register to 1 if the source register is not equal to zero.

    Equivalent to `sltu rd, x0, rs1 `
    """

    name = "riscv.snez"


@irdl_op_definition
class ZextBOp(RdRsIntegerOperation[IntRegisterType]):
    """
    A pseudo instruction that zero-extends the least-significant byte of the source to XLEN by copying the
    into all of the bits more significant than 31.

    Equivalent to `andi rd, rs1, 255`
    """

    name = "riscv.zext.b"

    traits = traits_def(Pure())


@irdl_op_definition
class ZextWOp(RdRsIntegerOperation[IntRegisterType]):
    """
    A pseudo instruction that zero-extends the least-significant word of the source to XLEN by inserting 0’s
    into all of the bits more significant than 31.

    Equivalent to `add.uw rd, rs1, 0`

    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-add_uw)
    """

    name = "riscv.zext.w"

    traits = traits_def(Pure())


@irdl_op_definition
class SextWOp(RdRsIntegerOperation[IntRegisterType]):
    """
    A pseudo instruction that writes the sign-extension of the lower 32 bits of register rs1 into register rd.

    Equivalent to `addiw rd, rs, 0 `

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/#_addiw).
    """

    name = "riscv.sext.w"

    traits = traits_def(Pure())


class FMVHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import RemoveRedundantFMv

        return (RemoveRedundantFMv(),)


@irdl_op_definition
class FMVOp(RdRsFloatOperation[FloatRegisterType]):
    """
    A pseudo instruction to copy contents of one float register to another.

    Equivalent to `fsgnj.s rd, rs, rs`.

    Both clang and gcc emit `fsw rs, 0(x); flw rd, 0(x)` to copy floats, possibly because
    storing and loading bits from memory is a lower overhead in practice than reasoning
    about floating-point values.
    """

    name = "riscv.fmv.s"

    traits = traits_def(
        Pure(),
        FMVHasCanonicalizationPatternsTrait(),
    )


## Integer Register-Register Operations


class AddOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            AddImmediates,
            AdditionOfSameVariablesToMultiplyByTwo,
        )

        return (AddImmediates(), AdditionOfSameVariablesToMultiplyByTwo())


@irdl_op_definition
class AddOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Adds the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    ```
    x[rd] = x[rs1] + x[rs2]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#add).
    """

    name = "riscv.add"

    traits = traits_def(
        Pure(),
        AddOpHasCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class SltOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Place the value 1 in register rd if register rs1 is less than register rs2 when both
    are treated as signed numbers, else 0 is written to rd.

    x[rd] = x[rs1] <s x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#slt).
    """

    name = "riscv.slt"


@irdl_op_definition
class SltuOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Place the value 1 in register rd if register rs1 is less than register rs2 when both
    are treated as unsigned numbers, else 0 is written to rd.

    x[rd] = x[rs1] <u x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sltu).
    """

    name = "riscv.sltu"


class BitwiseAndHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            BitwiseAndBySelf,
            BitwiseAndByZero,
        )

        return (BitwiseAndByZero(), BitwiseAndBySelf())


@irdl_op_definition
class AndOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs bitwise AND on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] & x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#and).
    """

    name = "riscv.and"

    traits = traits_def(BitwiseAndHasCanonicalizationPatternsTrait())


class BitwiseOrHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            BitwiseOrBySelf,
            BitwiseOrByZero,
        )

        return (BitwiseOrByZero(), BitwiseOrBySelf())


@irdl_op_definition
class OrOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs bitwise OR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] | x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#or).
    """

    name = "riscv.or"

    traits = traits_def(BitwiseOrHasCanonicalizationPatternsTrait())


class BitwiseXorHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            BitwiseXorByZero,
            XorBySelf,
        )

        return (XorBySelf(), BitwiseXorByZero())


@irdl_op_definition
class XorOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs bitwise XOR on registers rs1 and rs2 and place the result in rd.

    x[rd] = x[rs1] ^ x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#xor).
    """

    name = "riscv.xor"

    traits = traits_def(BitwiseXorHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SllOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs logical left shift on the value in register rs1 by the shift amount
    held in the lower 5 bits of register rs2.

    x[rd] = x[rs1] << x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sll).
    """

    name = "riscv.sll"


@irdl_op_definition
class SrlOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Logical right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2.

    x[rd] = x[rs1] >>u x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#srl).
    """

    name = "riscv.srl"


class SubOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            SubAddi,
            SubBySelf,
            SubImmediates,
        )

        return (SubImmediates(), SubAddi(), SubBySelf())


@irdl_op_definition
class SubOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Subtracts the registers rs1 and rs2 and stores the result in rd.
    Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.

    x[rd] = x[rs1] - x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub).
    """

    name = "riscv.sub"

    traits = traits_def(SubOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class SraOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs arithmetic right shift on the value in register rs1 by the shift amount held
    in the lower 5 bits of register rs2.

    x[rd] = x[rs1] >>s x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sub).
    """

    name = "riscv.sra"


@irdl_op_definition
class NopOp(NullaryOperation):
    """
    Does not change any user-visible state, except for advancing the pc register.
    Canonical nop is encoded as addi x0, x0, 0.
    """

    name = "riscv.nop"


# endregion

# region RV32I/RV64I: 2.5 Control Transfer Instructions

# Unconditional jumps


@irdl_op_definition
class JalOp(RdImmJumpOperation):
    """
    Jump to address and place return address in rd.

    jal mylabel is a pseudoinstruction for jal ra, mylabel

    x[rd] = pc+4; pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#jal).
    """

    name = "riscv.jal"


@irdl_op_definition
class JOp(RdImmJumpOperation):
    """
    A pseudo-instruction, for unconditional jumps you don't expect to return from.
    Is equivalent to JalOp with `rd` = `x0`.
    Used to be a part of the spec, removed in 2.0.
    """

    name = "riscv.j"

    def __init__(
        self,
        immediate: int | SImm20Attr | str | LabelAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        super().__init__(immediate, rd=Registers.ZERO, comment=comment)

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        # J op is a special case of JalOp with zero return register
        return (self.immediate,)


@irdl_op_definition
class JalrOp(RdRsImmJumpOperation):
    """
    Jump to address and place return address in rd.

    ```C
    t = pc+4
    pc = (x[rs1] + sext(offset)) & ~1
    x[rd] = t
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#jalr).
    """

    name = "riscv.jalr"


@irdl_op_definition
class ReturnOp(NullaryOperation):
    """
    Pseudo-op for returning from subroutine.

    Equivalent to `jalr x0, x1, 0`
    """

    name = "riscv.ret"

    traits = traits_def(IsTerminator())


# Conditional Branches


@irdl_op_definition
class BeqOp(RsRsOffIntegerOperation):
    """
    Take the branch if registers rs1 and rs2 are equal.

    ```C
    if (x[rs1] == x[rs2]) pc += sext(offset)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq).
    """

    name = "riscv.beq"


@irdl_op_definition
class BneOp(RsRsOffIntegerOperation):
    """
    Take the branch if registers rs1 and rs2 are not equal.

    ```C
    if (x[rs1] != x[rs2]) pc += sext(offset)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bne).
    """

    name = "riscv.bne"


@irdl_op_definition
class BltOp(RsRsOffIntegerOperation):
    """
    Take the branch if registers rs1 is less than rs2, using signed comparison.

    ```C
    if (x[rs1] <s x[rs2]) pc += sext(offset)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#blt).
    """

    name = "riscv.blt"


@irdl_op_definition
class BgeOp(RsRsOffIntegerOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using signed comparison.

    ```C
    if (x[rs1] >=s x[rs2]) pc += sext(offset)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bge).
    """

    name = "riscv.bge"


@irdl_op_definition
class BltuOp(RsRsOffIntegerOperation):
    """
    Take the branch if registers rs1 is less than rs2, using unsigned comparison.

    ```C
    if (x[rs1] <u x[rs2]) pc += sext(offset)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bltu).
    """

    name = "riscv.bltu"


@irdl_op_definition
class BgeuOp(RsRsOffIntegerOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using unsigned comparison.

    ```C
    if (x[rs1] >=u x[rs2]) pc += sext(offset)
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bgeu).
    """

    name = "riscv.bgeu"


# endregion

# region RV32I/RV64I: 2.6 Load and Store Instructions


@irdl_op_definition
class LbOp(RdRsImmIntegerOperation):
    """
    Loads a 8-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = sext(M[x[rs1] + sext(offset)][7:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lb).
    """

    name = "riscv.lb"


@irdl_op_definition
class LbuOp(RdRsImmIntegerOperation):
    """
    Loads a 8-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = M[x[rs1] + sext(offset)][7:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lbu).
    """

    name = "riscv.lbu"


@irdl_op_definition
class LhOp(RdRsImmIntegerOperation):
    """
    Loads a 16-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = sext(M[x[rs1] + sext(offset)][15:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lh).
    """

    name = "riscv.lh"


@irdl_op_definition
class LhuOp(RdRsImmIntegerOperation):
    """
    Loads a 16-bit value from memory and zero-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = M[x[rs1] + sext(offset)][15:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lhu).
    """

    name = "riscv.lhu"


class LwOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            LoadWordWithKnownOffset,
        )

        return (LoadWordWithKnownOffset(),)


@irdl_op_definition
class LwOp(RdRsImmIntegerOperation):
    """
    Loads a 32-bit value from memory and sign-extends this to XLEN bits before
    storing it in register rd.

    ```C
    x[rd] = sext(M[x[rs1] + sext(offset)][31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#lw).
    """

    name = "riscv.lw"

    traits = traits_def(LwOpHasCanonicalizationPatternTrait())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = _assembly_arg_str(self.rd)
        imm = _assembly_arg_str(self.immediate)
        offset = _assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({offset})", self.comment
        )


@irdl_op_definition
class SbOp(RsRsImmIntegerOperation):
    """
    Store 8-bit, values from the low bits of register rs2 to memory.

    ```C
    M[x[rs1] + sext(offset)] = x[rs2][7:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sb).
    """

    name = "riscv.sb"


@irdl_op_definition
class ShOp(RsRsImmIntegerOperation):
    """
    Store 16-bit, values from the low bits of register rs2 to memory.

    ```C
    M[x[rs1] + sext(offset)] = x[rs2][15:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sh).

    """

    name = "riscv.sh"


class SwOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            StoreWordWithKnownOffset,
        )

        return (StoreWordWithKnownOffset(),)


@irdl_op_definition
class SwOp(RsRsImmIntegerOperation):
    """
    Store 32-bit, values from the low bits of register rs2 to memory.

    ```C
    M[x[rs1] + sext(offset)] = x[rs2][31:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#sw).
    """

    name = "riscv.sw"

    traits = traits_def(SwOpHasCanonicalizationPatternTrait())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = _assembly_arg_str(self.rs2)
        imm = _assembly_arg_str(self.immediate)
        offset = _assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({offset})", self.comment
        )


# endregion

# region RV32I/RV64I: 2.8 Control and Status Register Instructions


@irdl_op_definition
class CsrrwOp(CsrReadWriteOperation):
    """
    Atomically swaps values in the CSRs and integer registers.
    CSRRW reads the old value of the CSR, zero-extends the value to XLEN bits,
    then writes it to integer register rd. The initial value in rs1 is written
    to the CSR. If the 'writeonly' attribute evaluates to False, then the
    instruction shall not read the CSR and shall not cause any of the side effects
    that might occur on a CSR read; in this case rd *must be allocated to x0*.

    t = CSRs[csr]; CSRs[csr] = x[rs1]; x[rd] = t

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrw).
    """

    name = "riscv.csrrw"


@irdl_op_definition
class CsrrsOp(CsrBitwiseOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd. The initial value in integer register rs1 is treated
    as a bit mask that specifies bit positions to be set in the CSR.
    Any bit that is high in rs1 will cause the corresponding bit to be set in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'readonly' attribute evaluates to True, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs. Note that if rs1 specifies a register holding a zero value other than x0,
    the instruction will still attempt to write the unmodified value back to the CSR and will
    cause any attendant side effects.

    t = CSRs[csr]; CSRs[csr] = t | x[rs1]; x[rd] = t

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrs).
    """

    name = "riscv.csrrs"


@irdl_op_definition
class CsrrcOp(CsrBitwiseOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd. The initial value in integer register rs1 is treated
    as a bit mask that specifies bit positions to be cleared in the CSR.
    Any bit that is high in rs1 will cause the corresponding bit to be cleared in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'readonly' attribute evaluates to True, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs. Note that if rs1 specifies a register holding a zero value other than x0,
    the instruction will still attempt to write the unmodified value back to the CSR and will
    cause any attendant side effects.

    t = CSRs[csr]; CSRs[csr] = t &~x[rs1]; x[rd] = t

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrc).
    """

    name = "riscv.csrrc"


@irdl_op_definition
class CsrrwiOp(CsrReadWriteImmOperation):
    """
    Update the CSR using an XLEN-bit value obtained by zero-extending the
    'immediate' attribute.
    If the 'writeonly' attribute evaluates to False, then the
    instruction shall not read the CSR and shall not cause any of the side effects
    that might occur on a CSR read; in this case rd *must be allocated to x0*.

    x[rd] = CSRs[csr]; CSRs[csr] = zimm

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrwi).
    """

    name = "riscv.csrrwi"


@irdl_op_definition
class CsrrsiOp(CsrBitwiseImmOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd. The value in the 'immediate' attribute is treated
    as a bit mask that specifies bit positions to be set in the CSR.
    Any bit that is high in it will cause the corresponding bit to be set in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'immediate' attribute value is zero, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs.

    t = CSRs[csr]; CSRs[csr] = t | zimm; x[rd] = t

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrsi).
    """

    name = "riscv.csrrsi"


@irdl_op_definition
class CsrrciOp(CsrBitwiseImmOperation):
    """
    Reads the value of the CSR, zero-extends the value to XLEN bits, and writes
    it to integer register rd.  The value in the 'immediate' attribute is treated
    as a bit mask that specifies bit positions to be cleared in the CSR.
    Any bit that is high in rs1 will cause the corresponding bit to be cleared in the CSR,
    if that CSR bit is writable. Other bits in the CSR are unaffected (though CSRs might
    have side effects when written).

    If the 'immediate' attribute value is zero, then the instruction will not write
    to the CSR at all, and so shall not cause any of the side effects that might otherwise
    occur on a CSR write, such as raising illegal instruction exceptions on accesses to
    read-only CSRs.

    t = CSRs[csr]; CSRs[csr] = t &~zimm; x[rd] = t

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#csrrci).
    """

    name = "riscv.csrrci"


# endregion

# region RV32M/RV64M: 7 “M” Standard Extension for Integer Multiplication and Division

## Multiplication Operations


class MulOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            MultiplyImmediates,
        )

        return (MultiplyImmediates(),)


@irdl_op_definition
class MulOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of signed rs1 by signed rs2
    and places the lower XLEN bits in the destination register.
    x[rd] = x[rs1] * x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#add).
    """

    name = "riscv.mul"

    traits = traits_def(MulOpHasCanonicalizationPatternsTrait(), Pure())


@irdl_op_definition
class MulhOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of signed rs1 by signed rs2
    and places the upper XLEN bits in the destination register.
    x[rd] = (x[rs1] s×s x[rs2]) >>s XLEN

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulh).
    """

    name = "riscv.mulh"


@irdl_op_definition
class MulhsuOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of signed rs1 by unsigned rs2
    and places the upper XLEN bits in the destination register.
    x[rd] = (x[rs1] s × x[rs2]) >>s XLEN

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulhsu).
    """

    name = "riscv.mulhsu"


@irdl_op_definition
class MulhuOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs an XLEN-bit × XLEN-bit multiplication of unsigned rs1 by unsigned rs2
    and places the upper XLEN bits in the destination register.
    x[rd] = (x[rs1] u × x[rs2]) >>u XLEN

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulhu).
    """

    name = "riscv.mulhu"


@irdl_op_definition
class MulwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs an 32-bit × 32-bit multiplication of signed rs1 by signed rs2.
    ```
    x[rd] = (x[rs1] s × x[rs2]) >>s XLEN
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#mulw).
    """

    name = "riscv.mulw"


class DivOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            DivideByOneIdentity,
        )

        return (DivideByOneIdentity(),)


## Division Operations
@irdl_op_definition
class DivOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an XLEN bits by XLEN bits signed integer division of rs1 by rs2,
    rounding towards zero.
    x[rd] = x[rs1] /s x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#div).
    """

    name = "riscv.div"
    traits = traits_def(DivOpHasCanonicalizationPatternsTrait(), Pure())


@irdl_op_definition
class DivuOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an XLEN bits by XLEN bits unsigned integer division of rs1 by rs2,
    rounding towards zero.
    x[rd] = x[rs1] /u x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#divu).
    """

    name = "riscv.divu"


@irdl_op_definition
class DivuwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an 32 bits by 32 bits unsigned integer division of rs1 by rs2.
    ```
    x[rd] = sext(x[rs1][31:0] /u x[rs2][31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64m.html#divuw).
    """

    name = "riscv.divuw"


@irdl_op_definition
class DivwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an 32 bits by 32 bits signed integer division of rs1 by rs2.
    ```
    x[rd] = sext(x[rs1][31:0] /s x[rs2][31:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#divw).
    """

    name = "riscv.divw"


@irdl_op_definition
class RemOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an XLEN bits by XLEN bits signed integer reminder of rs1 by rs2.
    x[rd] = x[rs1] %s x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#rem).
    """

    name = "riscv.rem"


@irdl_op_definition
class RemuOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an XLEN bits by XLEN bits unsigned integer reminder of rs1 by rs2.
    x[rd] = x[rs1] %u x[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvm.html#remu).
    """

    name = "riscv.remu"


@irdl_op_definition
class RemuwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an 32 bits by 32 bits unsigned integer reminder of rs1 by rs2.
    ```
    x[rd] = sext(x[rs1][31:0] %u x[rs2][31:0])
    ```
    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64m.html#remuw).
    """

    name = "riscv.remuw"


@irdl_op_definition
class RemwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Perform an 32 bits by 32 bits signed integer reminder of rs1 by rs2.
    ```
    x[rd] = sext(x[rs1][31:0] %s x[rs2][31:0])
    ```
    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rv64m.html#remw).
    """

    name = "riscv.remw"


# endregion

# region RV32B/RV64B: "B" Extension for Bit Manipulation, Version 1.0.0

## ZBB extension for Basic Bit-Manipulation. (not complete: population count missing)


@irdl_op_definition
class RolOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs a rotate left of rs1 by the amount in least-significant log2(XLEN) bits of rs2.
    ```
    let shamt = if   xlen == 32
                    then x[rs2][4..0]
                    else x[rs2][5..0];
    let result = (x[rs1] << shamt) | (x[rs2] >> (xlen - shamt));
    x[rd] = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-rol).
    """

    name = "riscv.rol"

    traits = traits_def(Pure())


@irdl_op_definition
class RorOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Performs a rotate right of rs1 by the amount in least-significant log2(XLEN) bits of rs2.
    ```
    let shamt = if   xlen == 32
                then x[rs2][4..0]
                else x[rs2][5..0];
    let result = (x[rs1] >> shamt) | (x[rs2] << (xlen - shamt));
    x[rd] = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-ror).
    """

    name = "riscv.ror"

    traits = traits_def(Pure())


@irdl_op_definition
class SextHOp(RdRsIntegerOperation[IntRegisterType]):
    """
    This instruction sign-extends the least-significant halfword in rs to XLEN by copying the
    most-significant bit in the halfword (i.e., bit 15) to all of the more-significant bits.
    ```
    x[rd] = EXTS(x[rs][15..0]);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sext_h).
    """

    name = "riscv.sext.h"

    traits = traits_def(Pure())


@irdl_op_definition
class ZextHOp(RdRsIntegerOperation[IntRegisterType]):
    """
    This instruction zero-extends the least-significant halfword of the source to XLEN by inserting
    0’s into all of the bits more significant than 15.
    ```
    x[rd] = EXTZ(x[rs][15..0]);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-zext_h).
    """

    name = "riscv.zext.h"

    traits = traits_def(Pure())


@irdl_op_definition
class SextBOp(RdRsIntegerOperation[IntRegisterType]):
    """
    This instruction sign-extends the least-significant byte in the source to XLEN by copying
    the most-significant bit in the byte (i.e., bit 7) to all of the more-significant bits.
    ```
    X[rd] = EXTS(X[rs][7..0]);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sext_b).
    """

    name = "riscv.sext.b"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of rs2.
    ```
    let index = X(rs2) & (XLEN - 1);
    X(rd) = X(rs1) & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bclr).
    """

    name = "riscv.bclr"

    traits = traits_def(Pure())


@irdl_op_definition
class BclrIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt.
    The index is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = X(rs1) & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bclri).
    """

    name = "riscv.bclri"

    traits = traits_def(Pure())


@irdl_op_definition
class BextOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns a single bit extracted from rs1 at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of rs2.
    ```
    let index = X(rs2) & (XLEN - 1);
    X(rd) = (X(rs1) >> index) & 1;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bext).
    """

    name = "riscv.bext"

    traits = traits_def(Pure())


@irdl_op_definition
class BextIOp(RdRsImmShiftOperation):
    """
    This instruction returns a single bit extracted from rs1 at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = (X(rs1) >> index) & 1;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bexti).
    """

    name = "riscv.bexti"

    traits = traits_def(Pure())


@irdl_op_definition
class BinvOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns rs1 with a single bit inverted at the index specified in shamt.
    The index is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings
    corresponding to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    X(rd) = X(rs1) ^ (1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-binvi).
    """

    name = "riscv.binv"

    traits = traits_def(Pure())


@irdl_op_definition
class BinvIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit cleared at the index specified in shamt. The index
    is read from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    x[rd] = x[rs1] & ~(1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-binvi).
    """

    name = "riscv.binvi"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns rs1 with a single bit set at the index specified in rs2.
    The index is read from the lower log2(XLEN) bits of rs2.
    ```
    let index = X(rs2) & (XLEN - 1);
    X(rd) = X(rs1) | (1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bset).
    """

    name = "riscv.bset"

    traits = traits_def(Pure())


@irdl_op_definition
class BsetIOp(RdRsImmShiftOperation):
    """
    This instruction returns rs1 with a single bit set at the index specified in shamt. The index is read
    from the lower log2(XLEN) bits of shamt. For RV32, the encodings corresponding
    to shamt[5]=1 are reserved.
    ```
    let index = shamt & (XLEN - 1);
    x[rd] = x[rs1] | (1 << index)
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-bseti).
    """

    name = "riscv.bseti"

    traits = traits_def(Pure())


@irdl_op_definition
class RolwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs a rotate left on the least-significant word of rs1 by the amount in
    least-significant 5 bits of rs2. The resulting word value is sign-extended by copying bit 31
    to all of the more-significant bits.
    ```
    let rs1 = EXTZ(X(rs1)[31..0])
    let shamt = X(rs2)[4..0];
    let result = (rs1 << shamt) | (rs1 >> (32 - shamt));
    X(rd) = EXTS(result);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-rolw).
    """

    name = "riscv.rolw"

    traits = traits_def(Pure())


@irdl_op_definition
class RorwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs a rotate right on the least-significant word of rs1 by the amount in
    least-significant 5 bits of rs2. The resultant word is sign-extended by copying bit 31 to all of
    the more-significant bits.
    ```
    let rs1 = EXTZ(X(rs1)[31..0])
    let shamt = X(rs2)[4..0];
    let result = (rs1 >> shamt) | (rs1 << (32 - shamt));
    X(rd) = EXTS(result);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-rorw).
    """

    name = "riscv.rorw"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriOp(RdRsImmShiftOperation):
    """
    This instruction performs a rotate right of rs1 by the amount in the least-significant
    log2(XLEN) bits of shamt. For RV32, the encodings corresponding to shamt[5]=1 are reserved.
    ```
    let shamt = if   xlen == 32
                    then shamt[4..0]
                    else shamt[5..0];
    let result = (X(rs1) >> shamt) | (X(rs2) << (xlen - shamt));
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-rori).
    """

    name = "riscv.rori"

    traits = traits_def(Pure())


@irdl_op_definition
class RoriwOp(RdRsImmShiftOperation):
    """
    This instruction performs a rotate right on the least-significant word of rs1 by the amount in
    the least-significant log2(XLEN) bits of shamt. The resulting word value is sign-extended by
    copying bit 31 to all of the more-significant bits.
    ```
    let rs1 = EXTZ(X(rs1)[31..0];
    let result = (rs1 >> shamt[4..0]) | (X(rs1) << (32 - shamt[4..0]));
    X(rd) = EXTS(result[31..0]);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-roriw).
    """

    name = "riscv.roriw"

    traits = traits_def(Pure())


@irdl_op_definition
class AddUwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs an XLEN-wide addition between rs2 and the zero-extended least-significant
    word of rs1.
    ```
    let base = X(rs2);
    let index = EXTZ(X(rs1)[31..0]);
    X(rd) = base + index;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-add_uw).
    """

    name = "riscv.add.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class Sh1addOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction shifts rs1 to the left by 1 bit and adds it to rs2.
    ```
    X(rd) = X(rs2) + (X(rs1) << 1);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sh1add).
    """

    name = "riscv.sh1add"

    traits = traits_def(Pure())


@irdl_op_definition
class Sh2addOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction shifts rs1 to the left by 2 places and adds it to rs2.
    ```
    X(rd) = X(rs2) + (X(rs1) << 2);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sh2add).
    """

    name = "riscv.sh2add"

    traits = traits_def(Pure())


@irdl_op_definition
class Sh3addOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction shifts rs1 to the left by 2 places and adds it to rs2.
    ```
    X(rd) = X(rs2) + (X(rs1) << 3);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sh3add).
    """

    name = "riscv.sh3add"

    traits = traits_def(Pure())


@irdl_op_definition
class Sh1addUwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs an XLEN-wide addition of two addends. The first addend is rs2.
    The second addend is the unsigned value formed by extracting the least-significant word of
    rs1 and shifting it left by 1 place.

    ```
    let base = x[rs2];
    let index = EXTZ(x[rs1][31..0]);
    x[rd] = base + (index << 1);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sh1add_uw).
    """

    name = "riscv.sh1add.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class Sh2addUwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs an XLEN-wide addition of two addends. The first addend is rs2.
    The second addend is the unsigned value formed by extracting the least-significant word of rs1
    and shifting it left by 2 places.
    ```
    let base = x[rs2];
    let index = EXTZ(x[rs1][31..0]);
    x[rd] = base + (index << 2);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sh2add_uw).
    """

    name = "riscv.sh2add.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class Sh3addUwOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs an XLEN-wide addition of two addends. The first addend is rs2.
    The second addend is the unsigned value formed by extracting the least-significant word of rs1
    and shifting it left by 3 places.

    ```
    let base = x[rs2];
    let index = EXTZ(x[rs1][31..0]);
    x[rd] = base + (index << 3);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-sh3add_uw).
    """

    name = "riscv.sh3add.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class SlliUwOp(RdRsImmShiftOperation):
    """
    This instruction takes the least-significant word of rs1, zero-extends it,
    and shifts it left by the immediate.
    ```
    x[rd] = (EXTZ(x[rs][31..0]) << shamt);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-slli_uw).
    """

    name = "riscv.slli.uw"

    traits = traits_def(Pure())


@irdl_op_definition
class AndnOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs the bitwise logical AND operation between rs1 and the bitwise inversion of rs2.
    ```
    X(rd) = X(rs1) & ~X(rs2);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-andn).
    """

    name = "riscv.andn"

    traits = traits_def(Pure())


@irdl_op_definition
class OrnOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs the bitwise logical OR operation between rs1 and the bitwise inversion of rs2.
    ```
    X(rd) = X(rs1) | ~X(rs2);
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-orn).
    """

    name = "riscv.orn"

    traits = traits_def(Pure())


@irdl_op_definition
class XnorOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction performs the bit-wise exclusive-NOR operation on rs1 and rs2.
    ```
    X(rd) = ~(X(rs1) ^ X(rs2));
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-xnor).
    """

    name = "riscv.xnor"

    traits = traits_def(Pure())


@irdl_op_definition
class MaxOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns the larger of two signed integers.
    ```
    let rs1_val = X(rs1);
    let rs2_val = X(rs2);

    let result = if   rs1_val <_s rs2_val
                    then rs2_val
                    else rs1_val;
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-max).
    """

    name = "riscv.max"

    traits = traits_def(Pure())


@irdl_op_definition
class MaxUOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns the larger of two unsigned integers.
    ```
    let rs1_val = X(rs1);
    let rs2_val = X(rs2);
    let result = if   rs1_val <_u rs2_val
                 then rs2_val
             else rs1_val;
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-maxu).
    """

    name = "riscv.maxu"

    traits = traits_def(Pure())


@irdl_op_definition
class MinOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns the smaller of two signed integers.
    ```
    let rs1_val = X(rs1);
    let rs2_val = X(rs2);
    let result = if   rs1_val <_s rs2_val
                 then rs1_val
             else rs2_val;
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-min).
    """

    name = "riscv.min"

    traits = traits_def(Pure())


@irdl_op_definition
class MinUOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    This instruction returns the smaller of two unsigned integers.
    ```
    let rs1_val = X(rs1);
    let rs2_val = X(rs2);
    let result = if   rs1_val <_u rs2_val
                    then rs1_val
                    else rs2_val;
    X(rd) = result;
    ```
    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#insns-minu).
    """

    name = "riscv.minu"

    traits = traits_def(Pure())


# endregion


# region "ZiCond" Conditional" operations extension
@irdl_op_definition
class CZeroEqzOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Moves zero to a register rd, if the condition rs2 is equal to zero, otherwise moves rs1 to rd.

    See external [documentation](https://github.com/riscvarchive/riscv-zicond/blob/main/zicondops.adoc).
    """

    name = "riscv.czero.eqz"


@irdl_op_definition
class CZeroNezOp(RdRsRsIntegerOperation[IntRegisterType, IntRegisterType]):
    """
    Moves zero to a register rd, if the condition rs2 is nonzero, otherwise moves rs1 to rd.

    See external [documentation](https://github.com/riscvarchive/riscv-zicond/blob/main/zicondops.adoc).
    """

    name = "riscv.czero.nez"


# endregion

# region Assembler pseudo-instructions
# See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md).


class LiOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            LoadImmediate0,
        )

        return (LoadImmediate0(),)


@irdl_op_definition
class LiOp(RISCVCustomFormatOperation, RISCVInstruction, ConstantLikeInterface, ABC):
    """
    Loads a 32-bit immediate into rd.

    This is an assembler pseudo-instruction.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#load-immediate).
    """

    name = "riscv.li"

    rd = result_def(IntRegisterType)
    immediate = attr_def(base(Imm32Attr) | base(LabelAttr))

    traits = traits_def(Pure(), LiOpHasCanonicalizationPatternTrait())

    def __init__(
        self,
        immediate: int | Imm32Attr | str | LabelAttr,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i32)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            result_types=[rd],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.immediate

    def get_constant_value(self):
        return self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i32)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        print_immediate_value(printer, self.immediate)
        return {"immediate", "fastmath"}

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        return (), (res_type,)

    def print_op_type(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.rd.type)


@irdl_op_definition
class EcallOp(NullaryOperation):
    """
    The ECALL instruction is used to make a request to the supporting execution
    environment, which is usually an operating system.
    The ABI for the system will define how parameters for the environment
    request are passed, but usually these will be in defined locations in the
    integer register file.

    See external [documentation](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf).
    """

    name = "riscv.ecall"


@irdl_op_definition
class LabelOp(RISCVCustomFormatOperation, RISCVAsmOperation):
    """
    The label operation is used to emit text labels (e.g. loop:) that are used
    as branch, unconditional jump targets and symbol offsets.

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#labels).
    """

    name = "riscv.label"
    label = attr_def(LabelAttr)
    comment = opt_attr_def(StringAttr)

    def __init__(
        self,
        label: str | LabelAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(label, str):
            label = LabelAttr(label)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            attributes={
                "label": label,
                "comment": comment,
            },
        )

    def assembly_line(self) -> str | None:
        return AssemblyPrinter.append_comment(f"{self.label.data}:", self.comment)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["label"] = LabelAttr(parser.parse_str_literal("Expected label"))
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        printer.print_string_literal(self.label.data)
        return {"label"}

    def print_op_type(self, printer: Printer) -> None:
        return

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        return (), ()


@irdl_op_definition
class DirectiveOp(RISCVCustomFormatOperation, RISCVAsmOperation):
    """
    The directive operation is used to emit assembler directives (e.g. .word; .equ; etc.)
    without any associated region of assembly code.
    A more complete list of directives can be found here:

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#pseudo-ops).
    """

    name = "riscv.directive"
    directive = attr_def(StringAttr)
    value = opt_attr_def(StringAttr)

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
            }
        )

    def assembly_line(self) -> str | None:
        if self.value is not None and self.value.data:
            arg_str = _assembly_arg_str(self.value.data)
        else:
            arg_str = ""

        return AssemblyPrinter.assembly_line(
            self.directive.data, arg_str, is_indented=False
        )

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["directive"] = StringAttr(
            parser.parse_str_literal("Expected directive")
        )
        if (value := parser.parse_optional_str_literal()) is not None:
            attributes["value"] = StringAttr(value)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        printer.print_string_literal(self.directive.data)
        if self.value is not None:
            printer.print_string(" ")
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
class AssemblySectionOp(IRDLOperation, AssemblyPrintable):
    """
    The directive operation is used to emit assembler directives (e.g. .text; .data; etc.)
    with the scope of a section.

    A more complete list of directives can be found here:

    See external [documentation](https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#pseudo-ops).

    This operation can have nested operations, corresponding to a section of the assembly.
    """

    name = "riscv.assembly_section"
    directive = attr_def(StringAttr)
    data = region_def("single_block")

    traits = traits_def(NoTerminator(), IsolatedFromAbove())

    def __init__(
        self,
        directive: str | StringAttr,
        region: Region | None = None,
    ):
        if isinstance(directive, str):
            directive = StringAttr(directive)
        if region is None:
            region = Region(Block())

        super().__init__(
            regions=[region],
            attributes={
                "directive": directive,
            },
        )

    @classmethod
    def parse(cls, parser: Parser) -> AssemblySectionOp:
        directive = parser.parse_str_literal()
        attr_dict = parser.parse_optional_attr_dict_with_keyword(("directive",))
        region = parser.parse_optional_region()

        if region is None:
            region = Region(Block())
        section = AssemblySectionOp(directive, region)
        if attr_dict is not None:
            section.attributes |= attr_dict.data

        return section

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_string_literal(self.directive.data)
        printer.print_op_attributes(
            self.attributes, reserved_attr_names=("directive",), print_keyword=True
        )
        printer.print_string(" ")
        if self.data.block.ops:
            printer.print_region(self.data)

    def print_assembly(self, printer: AssemblyPrinter) -> None:
        printer.emit_section(self.directive.data)


@irdl_op_definition
class CustomAssemblyInstructionOp(RISCVCustomFormatOperation, RISCVInstruction):
    """
    An instruction with unspecified semantics, that can be printed during assembly
    emission.

    During assembly emission, the results are printed before the operands:

    ``` python
    s0 = riscv.GetRegisterOp(Registers.s0).res
    s1 = riscv.GetRegisterOp(Registers.s1).res
    rs2 = riscv.Registers.s2
    rs3 = riscv.Registers.s3
    op = CustomAssemblyInstructionOp("my_instr", (s0, s1), (rs2, rs3))

    op.assembly_line()   # "my_instr s2, s3, s0, s1"
    ```
    """

    name = "riscv.custom_assembly_instruction"
    inputs = var_operand_def()
    outputs = var_result_def()
    instruction_name = attr_def(StringAttr)
    comment = opt_attr_def(StringAttr)

    def __init__(
        self,
        instruction_name: str | StringAttr,
        inputs: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(instruction_name, str):
            instruction_name = StringAttr(instruction_name)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[inputs],
            result_types=[result_types],
            attributes={
                "instruction_name": instruction_name,
                "comment": comment,
            },
        )

    def assembly_instruction_name(self) -> str:
        return self.instruction_name.data

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return *self.results, *self.operands


@irdl_op_definition
class CommentOp(RISCVCustomFormatOperation, RISCVAsmOperation):
    name = "riscv.comment"
    comment = attr_def(StringAttr)

    def __init__(self, comment: str | StringAttr):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            attributes={
                "comment": comment,
            },
        )

    def assembly_line(self) -> str | None:
        return f"    # {self.comment.data}"


@irdl_op_definition
class EbreakOp(NullaryOperation):
    """
    The EBREAK instruction is used by debuggers to cause control to be
    transferred back to a debugging environment.

    See external [documentation](https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf).
    """

    name = "riscv.ebreak"


@irdl_op_definition
class WfiOp(NullaryOperation):
    """
    The Wait for Interrupt instruction (WFI) provides a hint to the
    implementation that the current hart can be stalled until an
    interrupt might need servicing.

    See external [documentation](https://github.com/riscv/riscv-isa-manual/releases/download/Priv-v1.12/riscv-privileged-20211203.pdf).
    """

    name = "riscv.wfi"


# endregion

# region RISC-V SSA Helpers


class GetAnyRegisterOperation(
    RISCVCustomFormatOperation, RISCVAsmOperation, ABC, Generic[RDInvT]
):
    """
    This instruction allows us to create an SSAValue with for a given register name. This
    is useful for bridging the RISC-V convention that stores the result of function calls
    in `a0` and `a1` into SSA form.

    For example, to generate this assembly:
    ```
    jal my_func
    add a0 s0 a0
    ```

    One needs to do the following:

    ``` python
    rhs = riscv.GetRegisterOp(Registers.s0).res
    riscv.JalOp("my_func")
    lhs = riscv.GetRegisterOp(Registers.A0).res
    sum = riscv.AddOp(lhs, rhs, Registers.A0).rd
    ```
    """

    res = result_def(RDInvT)

    traits = traits_def(Pure())

    def __init__(
        self,
        register_type: RDInvT,
    ):
        super().__init__(result_types=[register_type])

    def assembly_line(self) -> str | None:
        # Don't print assembly for creating a SSA value representing register
        return None

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        return (), (res_type,)

    def print_op_type(self, printer: Printer) -> None:
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[IntRegisterType]):
    name = "riscv.get_register"


@irdl_op_definition
class GetFloatRegisterOp(GetAnyRegisterOperation[FloatRegisterType]):
    name = "riscv.get_float_register"


# endregion

# region RV32F: 8 “F” Standard Extension for Single-Precision Floating-Point, Version 2.0


class RdRsRsRsFloatOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that take three
    floating-point input registers and a destination register,
    e.g: fused-multiply-add (FMA) instructions.
    """

    rd = result_def(FloatRegisterType)
    rs1 = operand_def(FloatRegisterType)
    rs2 = operand_def(FloatRegisterType)
    rs3 = operand_def(FloatRegisterType)

    traits = traits_def(RegisterAllocatedMemoryEffect())

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        rs3: Operation | SSAValue,
        *,
        rd: FloatRegisterType = Registers.UNALLOCATED_FLOAT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2, rs3],
            attributes={
                "comment": comment,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.rs2, self.rs3


class RdRsRsFloatFloatIntegerOperationWithFastMath(
    RISCVCustomFormatOperation, RISCVInstruction, ABC
):
    """
    A base class for RISC-V operations that have two source floating-point
    registers with an integer destination register, and can be annotated with fastmath flags.

    This is called R-Type in the RISC-V specification.
    """

    rd = result_def(IntRegisterType)
    rs1 = operand_def(FloatRegisterType)
    rs2 = operand_def(FloatRegisterType)
    fastmath = attr_def(FastMathFlagsAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        *,
        rd: IntRegisterType = Registers.UNALLOCATED_INT,
        fastmath: FastMathFlagsAttr = FastMathFlagsAttr("none"),
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "comment": comment,
                "fastmath": fastmath,
            },
            result_types=[rd],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.rs2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        fast = FastMathFlagsAttr("none")
        if parser.parse_optional_keyword("fastmath") is not None:
            fast = FastMathFlagsAttr(FastMathFlagsAttr.parse_parameter(parser))
        attributes["fastmath"] = fast
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        if self.fastmath != FastMathFlagsAttr("none"):
            printer.print_string(" fastmath")
            self.fastmath.print_parameter(printer)
        return {"fastmath"}


class RsRsImmFloatOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that have two source registers
    (one integer and one floating-point) and an immediate.
    """

    rs1 = operand_def(IntRegisterType)
    rs2 = operand_def(FloatRegisterType)
    immediate = attr_def(Imm12Attr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: int | Imm12Attr | str | LabelAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i12)
        elif isinstance(immediate, str):
            immediate = LabelAttr(immediate)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rs1, self.rs2, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = parse_immediate_value(parser, i12)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class RdRsImmFloatOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RV32Foperations that have one floating-point
    destination register, one source register and
    one immediate operand.
    """

    rd = result_def(FloatRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(base(Imm12Attr) | base(LabelAttr))

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | Imm12Attr | str | LabelAttr,
        *,
        rd: FloatRegisterType = Registers.UNALLOCATED_FLOAT,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, i12)
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
        attributes["immediate"] = parse_immediate_value(parser, i12)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class FMAddSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply addition.

    ```C
    f[rd] = f[rs1]×f[rs2]+f[rs3]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmadd-s).
    """

    name = "riscv.fmadd.s"


@irdl_op_definition
class FMSubSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply substraction.

    ```C
    f[rd] = f[rs1]×f[rs2]+f[rs3]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmsub-s).
    """

    name = "riscv.fmsub.s"


@irdl_op_definition
class FNMSubSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply substraction.

    ```C
    f[rd] = -f[rs1]×f[rs2]+f[rs3]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fnmsub-s).
    """

    name = "riscv.fnmsub.s"


@irdl_op_definition
class FNMAddSOp(RdRsRsRsFloatOperation):
    """
    Perform single-precision fused multiply addition.

    ```C
    f[rd] = -f[rs1]×f[rs2]-f[rs3]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fnmadd-s).
    """

    name = "riscv.fnmadd.s"


@irdl_op_definition
class FAddSOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform single-precision floating-point addition.

    ```C
    f[rd] = f[rs1]+f[rs2]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fadd-s).
    """

    name = "riscv.fadd.s"

    traits = traits_def(Pure())


@irdl_op_definition
class FSubSOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform single-precision floating-point substraction.

    ```C
    f[rd] = f[rs1]-f[rs2]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsub-s).
    """

    name = "riscv.fsub.s"


@irdl_op_definition
class FMulSOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform single-precision floating-point multiplication.

    ```C
    f[rd] = f[rs1]×f[rs2]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmul-s).
    """

    name = "riscv.fmul.s"


@irdl_op_definition
class FDivSOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform single-precision floating-point division.

    ```C
    f[rd] = f[rs1] / f[rs2]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fdiv-s).
    """

    name = "riscv.fdiv.s"


@irdl_op_definition
class FSqrtSOp(RdRsFloatOperation[FloatRegisterType]):
    """
    Perform single-precision floating-point square root.

    ```C
    f[rd] = sqrt(f[rs1])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsqrt-s).
    """

    name = "riscv.fsqrt.s"


@irdl_op_definition
class FSgnJSOp(RdRsRsFloatOperation[FloatRegisterType, FloatRegisterType]):
    """
    Produce a result that takes all bits except the sign bit from rs1.
    The result’s sign bit is rs2’s sign bit.

    ```C
    f[rd] = {f[rs2][31], f[rs1][30:0]}
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsgnj.s).
    """

    name = "riscv.fsgnj.s"


@irdl_op_definition
class FSgnJNSOp(RdRsRsFloatOperation[FloatRegisterType, FloatRegisterType]):
    """
    Produce a result that takes all bits except the sign bit from rs1.
    The result’s sign bit is opposite of rs2’s sign bit.

    ```C
    f[rd] = {~f[rs2][31], f[rs1][30:0]}
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsgnjn.s).
    """

    name = "riscv.fsgnjn.s"


@irdl_op_definition
class FSgnJXSOp(RdRsRsFloatOperation[FloatRegisterType, FloatRegisterType]):
    """
    Produce a result that takes all bits except the sign bit from rs1.
    The result’s sign bit is XOR of sign bit of rs1 and rs2.

    ```C
    f[rd] = {f[rs1][31] ^ f[rs2][31], f[rs1][30:0]}
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsgnjx.s).
    """

    name = "riscv.fsgnjx.s"


@irdl_op_definition
class FMinSOp(RdRsRsFloatOperationWithFastMath):
    """
    Write the smaller of single precision data in rs1 and rs2 to rd.

    ```C
    f[rd] = min(f[rs1], f[rs2])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmin-s).
    """

    name = "riscv.fmin.s"


@irdl_op_definition
class FMaxSOp(RdRsRsFloatOperationWithFastMath):
    """
    Write the larger of single precision data in rs1 and rs2 to rd.

    ```C
    f[rd] = max(f[rs1], f[rs2])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmax-s).
    """

    name = "riscv.fmax.s"


@irdl_op_definition
class FCvtWSOp(RdRsIntegerOperation[FloatRegisterType]):
    """
    Convert a floating-point number in floating-point register rs1 to a signed 32-bit in integer register rd.

    ```C
    x[rd] = sext(s32_{f32}(f[rs1]))
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.w.s).
    """

    name = "riscv.fcvt.w.s"


@irdl_op_definition
class FCvtWuSOp(RdRsIntegerOperation[FloatRegisterType]):
    """
    Convert a floating-point number in floating-point register rs1 to a signed 32-bit in unsigned integer register rd.

    ```C
    x[rd] = sext(u32_{f32}(f[rs1]))
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.wu.s).
    """

    name = "riscv.fcvt.wu.s"


@irdl_op_definition
class FMvXWOp(RdRsIntegerOperation[FloatRegisterType]):
    """
    Move the single-precision value in floating-point register rs1 represented in IEEE
    754-2008 encoding to the lower 32 bits of integer register rd.

    ```C
    x[rd] = sext(f[rs1][31:0])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmv.x.w).
    """

    name = "riscv.fmv.x.w"


@irdl_op_definition
class FeqSOp(RdRsRsFloatFloatIntegerOperationWithFastMath):
    """
    Performs a quiet equal comparison between floating-point registers rs1 and rs2 and
    record the Boolean result in integer register rd.
    Only signaling NaN inputs cause an Invalid Operation exception.
    The result is 0 if either operand is NaN.

    x[rd] = f[rs1] == f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#feq.s).
    """

    name = "riscv.feq.s"


@irdl_op_definition
class FltSOp(RdRsRsFloatFloatIntegerOperationWithFastMath):
    """
    Performs a quiet less comparison between floating-point registers rs1 and rs2 and
    record the Boolean result in integer register rd.
    Only signaling NaN inputs cause an Invalid Operation exception.
    The result is 0 if either operand is NaN.

    x[rd] = f[rs1] < f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#flt.s).
    """

    name = "riscv.flt.s"


@irdl_op_definition
class FleSOp(RdRsRsFloatFloatIntegerOperationWithFastMath):
    """
    Performs a quiet less or equal comparison between floating-point registers rs1 and
    rs2 and record the Boolean result in integer register rd.
    Only signaling NaN inputs cause an Invalid Operation exception.
    The result is 0 if either operand is NaN.

    x[rd] = f[rs1] <= f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fle.s).
    """

    name = "riscv.fle.s"


@irdl_op_definition
class FClassSOp(RdRsIntegerOperation[FloatRegisterType]):
    """
    Examines the value in floating-point register rs1 and writes to integer register rd
    a 10-bit mask that indicates the class of the floating-point number.
    The format of the mask is described in [classify table]_.
    The corresponding bit in rd will be set if the property is true and clear otherwise.
    All other bits in rd are cleared. Note that exactly one bit in rd will be set.

    x[rd] = classifys(f[rs1])

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fclass.s).
    """

    name = "riscv.fclass.s"


@irdl_op_definition
class FCvtSWOp(RdRsFloatOperation[IntRegisterType]):
    """
    Converts a 32-bit signed integer, in integer register rs1 into a floating-point number in floating-point register rd.

    ```C
    f[rd] = f32_{s32}(x[rs1])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.s.w).
    """

    name = "riscv.fcvt.s.w"


@irdl_op_definition
class FCvtSWuOp(RdRsFloatOperation[IntRegisterType]):
    """
    Converts a 32-bit unsigned integer, in integer register rs1 into a floating-point
    number in floating-point register rd.

    ```C
    f[rd] = f32_{u32}(x[rs1])
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt.s.wu).
    """

    name = "riscv.fcvt.s.wu"


@irdl_op_definition
class FMvWXOp(RdRsFloatOperation[IntRegisterType]):
    """
    Move the single-precision value encoded in IEEE 754-2008 standard encoding from the
    lower 32 bits of integer register rs1 to the floating-point register rd.

    ```C
    f[rd] = x[rs1][31:0]
    ```


    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmv.w.x).
    """

    name = "riscv.fmv.w.x"


class FLwOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            LoadFloatWordWithKnownOffset,
        )

        return (LoadFloatWordWithKnownOffset(),)


@irdl_op_definition
class FLwOp(RdRsImmFloatOperation):
    """
    Load a single-precision value from memory into floating-point register rd.

    ```C
    f[rd] = M[x[rs1] + sext(offset)][31:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#flw).
    """

    name = "riscv.flw"

    traits = traits_def(FLwOpHasCanonicalizationPatternTrait())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = _assembly_arg_str(self.rd)
        imm = _assembly_arg_str(self.immediate)
        offset = _assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({offset})", self.comment
        )


class FSwOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            StoreFloatWordWithKnownOffset,
        )

        return (StoreFloatWordWithKnownOffset(),)


@irdl_op_definition
class FSwOp(RsRsImmFloatOperation):
    """
    Store a single-precision value from floating-point register rs2 to memory.

    M[x[rs1] + offset] = f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsw).
    """

    name = "riscv.fsw"

    traits = traits_def(FSwOpHasCanonicalizationPatternTrait())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = _assembly_arg_str(self.rs2)
        imm = _assembly_arg_str(self.immediate)
        offset = _assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({offset})", self.comment
        )


# endregion

# region RV32F: 9 “D” Standard Extension for Double-Precision Floating-Point, Version 2.0


@irdl_op_definition
class FMAddDOp(RdRsRsRsFloatOperation):
    """
    Perform double-precision fused multiply addition.

    f[rd] = f[rs1]×f[rs2]+f[rs3]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmadd-d).
    """

    name = "riscv.fmadd.d"

    traits = traits_def(Pure())


@irdl_op_definition
class FMSubDOp(RdRsRsRsFloatOperation):
    """
    Perform double-precision fused multiply substraction.

    f[rd] = f[rs1]×f[rs2]+f[rs3]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmsub-d).
    """

    name = "riscv.fmsub.d"

    traits = traits_def(Pure())


class FuseMultiplyAddDCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            FuseMultiplyAddD,
        )

        return (FuseMultiplyAddD(),)


@irdl_op_definition
class FAddDOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform double-precision floating-point addition.

    f[rd] = f[rs1]+f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fadd-d).
    """

    name = "riscv.fadd.d"

    traits = traits_def(
        Pure(),
        FuseMultiplyAddDCanonicalizationPatternTrait(),
    )


@irdl_op_definition
class FSubDOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform double-precision floating-point substraction.

    f[rd] = f[rs1]-f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsub-d).
    """

    name = "riscv.fsub.d"

    traits = traits_def(Pure())


@irdl_op_definition
class FMulDOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform double-precision floating-point multiplication.

    f[rd] = f[rs1]×f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmul-d).
    """

    name = "riscv.fmul.d"

    traits = traits_def(Pure())


@irdl_op_definition
class FDivDOp(RdRsRsFloatOperationWithFastMath):
    """
    Perform double-precision floating-point division.

    f[rd] = f[rs1] / f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fdiv-d).
    """

    name = "riscv.fdiv.d"


class FLdOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            LoadDoubleWithKnownOffset,
        )

        return (LoadDoubleWithKnownOffset(),)


@irdl_op_definition
class FMinDOp(RdRsRsFloatOperationWithFastMath):
    """
    Write the smaller of double precision data in rs1 and rs2 to rd.

    f[rd] = min(f[rs1], f[rs2])

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmin-d).
    """

    name = "riscv.fmin.d"

    traits = traits_def(Pure())


@irdl_op_definition
class FMaxDOp(RdRsRsFloatOperationWithFastMath):
    """
    Write the larger of single precision data in rs1 and rs2 to rd.

    f[rd] = max(f[rs1], f[rs2])

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fmax-d).
    """

    name = "riscv.fmax.d"

    traits = traits_def(Pure())


@irdl_op_definition
class FCvtDWOp(RdRsFloatOperation[IntRegisterType]):
    """
    Converts a 32-bit signed integer, in integer register rs1 into a double-precision
    floating-point number in floating-point register rd.

    x[rd] = sext(s32_{f64}(f[rs1]))

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt-d-w).
    """

    name = "riscv.fcvt.d.w"

    traits = traits_def(Pure())


@irdl_op_definition
class FCvtDWuOp(RdRsFloatOperation[IntRegisterType]):
    """
    Converts a 32-bit unsigned integer, in integer register rs1 into a double-precision
    floating-point number in floating-point register rd.

    f[rd] = f64_{u32}(x[rs1])

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt-d-wu).
    """

    name = "riscv.fcvt.d.wu"

    traits = traits_def(Pure())


@irdl_op_definition
class FLdOp(RdRsImmFloatOperation):
    """
    Load a double-precision value from memory into floating-point register rd.

    ```C
    f[rd] = M[x[rs1] + sext(offset)][63:0]
    ```

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fld).
    """

    name = "riscv.fld"

    traits = traits_def(FLdOpHasCanonicalizationPatternTrait())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = _assembly_arg_str(self.rd)
        imm = _assembly_arg_str(self.immediate)
        offset = _assembly_arg_str(self.rs1)
        if isinstance(self.immediate, LabelAttr):
            return AssemblyPrinter.assembly_line(
                instruction_name, f"{value}, {imm}, {offset}", self.comment
            )
        else:
            return AssemblyPrinter.assembly_line(
                instruction_name, f"{value}, {imm}({offset})", self.comment
            )


class FSdOpHasCanonicalizationPatternTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            StoreDoubleWithKnownOffset,
        )

        return (StoreDoubleWithKnownOffset(),)


@irdl_op_definition
class FSdOp(RsRsImmFloatOperation):
    """
    Store a double-precision value from floating-point register rs2 to memory.

    M[x[rs1] + offset] = f[rs2]

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fsw).
    """

    name = "riscv.fsd"

    traits = traits_def(FSdOpHasCanonicalizationPatternTrait())

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        value = _assembly_arg_str(self.rs2)
        imm = _assembly_arg_str(self.immediate)
        offset = _assembly_arg_str(self.rs1)
        return AssemblyPrinter.assembly_line(
            instruction_name, f"{value}, {imm}({offset})", self.comment
        )


class FMvDHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import RemoveRedundantFMvD

        return (RemoveRedundantFMvD(),)


@irdl_op_definition
class FMvDOp(RdRsFloatOperation[FloatRegisterType]):
    """
    A pseudo instruction to copy 64 bits of one float register to another.

    Equivalent to `fsgnj.d rd, rs, rs`.
    """

    name = "riscv.fmv.d"

    traits = traits_def(
        Pure(),
        FMvDHasCanonicalizationPatternsTrait(),
    )


# endregion

# region 17 "V" Standard Extension for Vector Operations

# See external documentation https://riscv.org/wp-content/uploads/2018/05/15.20-15.55-18.05.06.VEXT-bcn-v1.pdf

# Vector operations that use standard RISC-V registers are using a non-standard Xfvec
# extension.
# All Xfvec instructions performing vectorial single precision operations require 64bit
# floating point registers (a.k.a.: FLEN==64).
# https://iis-git.ee.ethz.ch/smach/smallFloat-spec/-/raw/master/smallFloat_isa.pdf


@irdl_op_definition
class VFAddSOp(RdRsRsFloatOperation[FloatRegisterType, FloatRegisterType]):
    """
    Perform a pointwise single-precision floating-point addition over vectors.

    If the registers used are FloatRegisterType, they must be 64-bit wide, and contain two
    32-bit single-precision floating point values.
    """

    name = "riscv.vfadd.s"

    traits = traits_def(Pure())


@irdl_op_definition
class VFMulSOp(RdRsRsFloatOperation[FloatRegisterType, FloatRegisterType]):
    """
    Perform a pointwise single-precision floating-point multiplication over vectors.

    If the registers used are FloatRegisterType, they must be 64-bit wide, and contain two
    32-bit single-precision floating point values.
    """

    name = "riscv.vfmul.s"

    traits = traits_def(Pure())


# endregion


def _parse_optional_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr | None:
    """
    Parse an optional immediate value. If an integer is parsed, an integer attr with the specified type is created.
    """
    pos = parser.pos
    if (immediate := parser.parse_optional_integer()) is not None:
        try:
            return IntegerAttr(immediate, integer_type)
        except VerifyException as e:
            parser.raise_error(e.args[0], pos)
    if (immediate := parser.parse_optional_str_literal()) is not None:
        return LabelAttr(immediate)


def parse_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr:
    return parser.expect(
        lambda: _parse_optional_immediate_value(parser, integer_type),
        "Expected immediate",
    )


def print_immediate_value(printer: Printer, immediate: IntegerAttr | LabelAttr):
    match immediate:
        case IntegerAttr():
            immediate.print_without_type(printer)
        case LabelAttr():
            printer.print_string_literal(immediate.data)


RISCV = Dialect(
    "riscv",
    [
        AddiOp,
        SltiOp,
        SltiuOp,
        AndiOp,
        OriOp,
        XoriOp,
        SlliOp,
        SrliOp,
        SraiOp,
        LuiOp,
        AuipcOp,
        MVOp,
        SeqzOp,
        SnezOp,
        ZextBOp,
        ZextWOp,
        SextWOp,
        AddOp,
        SltOp,
        SltuOp,
        AndOp,
        OrOp,
        XorOp,
        SllOp,
        SrlOp,
        SubOp,
        SraOp,
        NopOp,
        JalOp,
        JOp,
        JalrOp,
        ReturnOp,
        BeqOp,
        BneOp,
        BltOp,
        BgeOp,
        BltuOp,
        BgeuOp,
        LbOp,
        LbuOp,
        LhOp,
        LhuOp,
        LwOp,
        SbOp,
        ShOp,
        SwOp,
        CsrrwOp,
        CsrrsOp,
        CsrrcOp,
        CsrrwiOp,
        CsrrsiOp,
        CsrrciOp,
        MulOp,
        MulhOp,
        MulhsuOp,
        MulhuOp,
        DivOp,
        DivuOp,
        RemOp,
        RemuOp,
        LiOp,
        RolOp,
        RorOp,
        RemuwOp,
        SrliwOp,
        SraiwOp,
        AddwOp,
        SubwOp,
        SllwOp,
        SrlwOp,
        SrawOp,
        RemwOp,
        MulwOp,
        DivwOp,
        DivuwOp,
        CZeroEqzOp,
        CZeroNezOp,
        BclrOp,
        BextOp,
        BinvOp,
        BsetOp,
        RolwOp,
        RorwOp,
        AddUwOp,
        Sh1addOp,
        Sh2addOp,
        Sh3addOp,
        Sh1addUwOp,
        Sh2addUwOp,
        Sh3addUwOp,
        SextBOp,
        SextHOp,
        ZextHOp,
        AndnOp,
        OrnOp,
        XnorOp,
        MaxOp,
        MaxUOp,
        MinOp,
        MinUOp,
        BclrIOp,
        BextIOp,
        BsetIOp,
        BinvIOp,
        RoriOp,
        RoriwOp,
        SlliUwOp,
        EcallOp,
        LabelOp,
        DirectiveOp,
        AssemblySectionOp,
        EbreakOp,
        WfiOp,
        CustomAssemblyInstructionOp,
        CommentOp,
        GetRegisterOp,
        GetFloatRegisterOp,
        # Floating point
        FMVOp,
        FMAddSOp,
        FMSubSOp,
        FNMSubSOp,
        FNMAddSOp,
        FAddSOp,
        FSubSOp,
        FMulSOp,
        FDivSOp,
        FSqrtSOp,
        FSgnJSOp,
        FSgnJNSOp,
        FSgnJXSOp,
        FMinSOp,
        FMaxSOp,
        FCvtWSOp,
        FCvtWuSOp,
        FMvXWOp,
        FeqSOp,
        FltSOp,
        FleSOp,
        FClassSOp,
        FCvtSWOp,
        FCvtSWuOp,
        FMvWXOp,
        FLwOp,
        FSwOp,
        FMAddDOp,
        FMSubDOp,
        FAddDOp,
        FSubDOp,
        FMulDOp,
        FDivDOp,
        FMinDOp,
        FMaxDOp,
        FCvtDWOp,
        FCvtDWuOp,
        FLdOp,
        FSdOp,
        FMvDOp,
        VFAddSOp,
        VFMulSOp,
    ],
    [
        IntRegisterType,
        FloatRegisterType,
        LabelAttr,
        FastMathFlagsAttr,
    ],
)
