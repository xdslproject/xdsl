from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from io import StringIO
from typing import IO, Generic, TypeAlias

from typing_extensions import Self

from xdsl.backend.assembly_printer import AssemblyPrinter, OneLineAssemblyPrintable
from xdsl.backend.register_allocatable import (
    HasRegisterConstraints,
    RegisterConstraints,
)
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect, RegisterType
from xdsl.dialects.builtin import (
    I32,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import HasCanonicalizationPatternsTrait, Pure
from xdsl.utils.exceptions import VerifyException

from .attrs import (
    I12,
    I20,
    SI12,
    SI20,
    UI5,
    FastMathFlagsAttr,
    LabelAttr,
    i12,
    i20,
    parse_immediate_value,
    print_immediate_value,
    si12,
    si20,
    ui5,
)
from .registers import (
    FloatRegisterType,
    IntRegisterType,
    RDInvT,
    Registers,
    RS1InvT,
    RS2InvT,
    RSInvT,
    is_non_zero,
)


class RISCVAsmOperation(IRDLOperation, OneLineAssemblyPrintable, ABC):
    """
    Base class for operations that can be a part of RISC-V assembly printing.
    """


class RISCVRegallocOperation(HasRegisterConstraints, IRDLOperation, ABC):
    """
    Base class for operations that can take part in register allocation.
    """

    def get_register_constraints(self) -> RegisterConstraints:
        # The default register constraints are that all operands are "in", and all
        # results are "out" registers.
        # If some registers are "inout" then this function must be overridden.
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


class RISCVInstruction(RISCVAsmOperation, RISCVRegallocOperation, ABC):
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
            assembly_arg_str(arg)
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return AssemblyPrinter.assembly_line(instruction_name, arg_str, self.comment)


# region Assembly printing


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
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

    rd: OpResult[RDInvT] = result_def(RDInvT)
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


class RsRsImmFloatOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RV32F operations that have two source registers
    (one integer and one floating-point) and an immediate.
    """

    rs1 = operand_def(IntRegisterType)
    rs2 = operand_def(FloatRegisterType)
    immediate = attr_def(IntegerAttr[I12])

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: int | IntegerAttr[I12] | str | LabelAttr,
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
    immediate = attr_def(IntegerAttr[I12] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[I12] | str | LabelAttr,
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
    immediate = attr_def(IntegerAttr[I20] | LabelAttr)

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
    immediate = attr_def(IntegerAttr[SI20] | LabelAttr)

    def __init__(
        self,
        immediate: int | IntegerAttr[SI20] | str | LabelAttr,
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
    immediate = attr_def(IntegerAttr[SI12] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[SI12] | str | LabelAttr,
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


class ImmShiftOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv import (
            ShiftbyZero,
            ShiftConstantFolding,
        )

        return (ShiftbyZero(), ShiftConstantFolding())


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
    immediate = attr_def(IntegerAttr[UI5] | LabelAttr)
    traits = traits_def(ImmShiftOpHasCanonicalizationPatternsTrait())

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[UI5] | str | LabelAttr,
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

    @abstractmethod
    def py_operation(self, rs1: IntegerAttr[I32]) -> IntegerAttr[I32] | None:
        """
        Performs a python function corresponding to this operation.

        If `i := py_operation(rs1)` is an IntegerAttr[I32], then this operation can be
        canonicalized to a constant with value `i` when the inputs are constants
        with values `rs1`. The immediate value is retrieved from the `immediate` attribute of the operation.
        """

        raise NotImplementedError(
            "RdRsImmShiftOperation py_operation is not yet implemented"
        )


class RdRsImmBitManipOperation(RISCVCustomFormatOperation, RISCVInstruction, ABC):
    """
    A base class for RISC-V operations that have one destination register, one source
    register and one immediate operand.

    These operations are from the Zba, Zbb, and Zbs extensions.

    The immediate value is encoded in the lower 5 bits of the immediate field for RV32
    and the lower 6 bits for RV64.

    See external [documentation](https://five-embeddev.com/riscv-bitmanip/1.0.0/bitmanip.html#).
    """

    rd = result_def(IntRegisterType)
    rs1 = operand_def(IntRegisterType)
    immediate = attr_def(IntegerAttr[UI5] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[UI5] | str | LabelAttr,
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
    immediate = attr_def(IntegerAttr[SI12] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        immediate: int | IntegerAttr[SI12] | str | LabelAttr,
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
    offset = attr_def(IntegerAttr[SI12] | LabelAttr)

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        offset: int | IntegerAttr[SI12] | LabelAttr,
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
    immediate = attr_def(IntegerAttr[SI12])

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        immediate: int | IntegerAttr[SI12] | str | LabelAttr,
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


class GetAnyRegisterOperation(
    RISCVCustomFormatOperation,
    RISCVAsmOperation,
    RISCVRegallocOperation,
    ABC,
    Generic[RDInvT],
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
    rhs = rv32.GetRegisterOp(Registers.s0).res
    riscv.JalOp("my_func")
    lhs = rv32.GetRegisterOp(Registers.A0).res
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


# endregion
