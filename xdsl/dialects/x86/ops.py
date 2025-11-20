"""
The `x86` dialect contains operations that represent x86 assembly operations.

In x86, the assembly operations may have different meaning depending on the types of
arguments.
For example, the [`mov` instruction](https://www.felixcloutier.com/x86/mov) can assign
an immediate value to a register, or move the contents of another register.
In order to disambiguate the two, we use a mnemonic in the operation name to communicate
which operands are expected, for example `x86.ds.mov` is the version that moves the
contents of one register to another, and `x86.di.mov` is the version that sets the
immediate value passed in to the register.
The mnemonic encodes the types of the assembly instruction arguments, in order.

Here are the possible mnemonic values and what they stand for:


- `s`: Source register
- `d`: Destination register
- `k`: Mask register
- `r`: Register used both as a source and destination
- `i`: Immediate value
- `m`: Memory
- `c`: Condition

This dialect is structured into abstract base classes, which are prefixed with the
mnemonic that corresponds to the subclassing operations (e.g. `DS_Operation`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from io import StringIO
from typing import IO, ClassVar, Generic, Literal, cast

from typing_extensions import Self, TypeVar

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
    Signedness,
    StringAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.ir import (
    Attribute,
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Successor,
    VarConstraint,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    IsTerminator,
    MemoryReadEffect,
    MemoryWriteEffect,
    Pure,
)
from xdsl.utils.exceptions import VerifyException

from .assembly import (
    AssemblyInstructionArg,
    assembly_arg_str,
    masked_source_str,
    memory_access_str,
    parse_immediate_value,
    parse_optional_immediate_value,
    parse_type_pair,
    print_immediate_value,
    print_type_pair,
)
from .attributes import LabelAttr
from .registers import (
    RAX,
    RDX,
    RFLAGS,
    RSP,
    AVX512MaskRegisterType,
    AVX512RegisterType,
    GeneralRegisterType,
    RFLAGSRegisterType,
    X86RegisterType,
    X86VectorRegisterType,
)

R1InvT = TypeVar("R1InvT", bound=X86RegisterType)
R2InvT = TypeVar("R2InvT", bound=X86RegisterType)
R3InvT = TypeVar("R3InvT", bound=X86RegisterType)
R4InvT = TypeVar("R4InvT", bound=X86RegisterType)


class X86AsmOperation(
    IRDLOperation, HasRegisterConstraints, OneLineAssemblyPrintable, ABC
):
    """
    Base class for operations that can be a part of x86 assembly printing.
    """

    traits = traits_def(RegisterAllocatedMemoryEffect())

    @abstractmethod
    def assembly_line(self) -> str | None:
        raise NotImplementedError()

    def iter_used_registers(self):
        return (
            val.type
            for vals in (self.operands, self.results)
            for val in vals
            if isinstance(val.type, RegisterType) and val.type.is_allocated
        )

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(self.operands, self.results, ())


class X86CustomFormatOperation(IRDLOperation, ABC):
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
        return cls.build(
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
        )

    @classmethod
    def parse_optional_memory_access_offset(
        cls, parser: Parser, integer_type: IntegerType = i64
    ) -> Attribute | None:
        return parse_optional_immediate_value(
            parser,
            integer_type,
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


class X86Instruction(X86AsmOperation):
    """
    Base class for operations that can be a part of x86 assembly printing. Must
    represent an instruction in the x86 instruction set.
    The name of the operation will be used as the x86 assembly instruction name.
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

        return self.name.split(".")[-1]

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            assembly_arg_str(arg)
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return AssemblyPrinter.assembly_line(instruction_name, arg_str, self.comment)


# region: Operation Base Classes


class RS_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that have one register that is read and written to,
    and one source register.
    """

    register_in = operand_def(R1InvT)
    register_out = result_def(R1InvT)

    source = operand_def(R2InvT)

    def __init__(
        self,
        register_in: Operation | SSAValue,
        source: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        register_in = SSAValue.get(register_in)
        if register_out is None:
            register_out = cast(R1InvT, register_in.type)

        super().__init__(
            operands=[register_in, source],
            attributes={
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.register_in, self.source

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source,), (), ((self.register_in, self.register_out),)
        )


class DS_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that have one destination register and one source
    register.
    """

    destination: OpResult[R1InvT] = result_def(R1InvT)
    source = operand_def(R2InvT)

    def __init__(
        self,
        source: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
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
        return (self.destination, self.source)


class DSK_Operation(X86Instruction, X86CustomFormatOperation, ABC):
    """
    A base class for x86 operations that have one destination register and one source
    register.
    """

    destination: OpResult[AVX512RegisterType] = result_def(AVX512RegisterType)
    source = operand_def(AVX512RegisterType)
    mask_reg = operand_def(AVX512MaskRegisterType)
    z = opt_attr_def(UnitAttr)

    def __init__(
        self,
        source: Operation | SSAValue,
        mask_reg: Operation | SSAValue,
        *,
        z: bool = False,
        comment: str | StringAttr | None = None,
        destination: AVX512RegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source, mask_reg],
            attributes={
                "z": UnitAttr() if z else None,
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        register_out = masked_source_str(self.destination, self.mask_reg, self.z)
        return register_out, self.source


class R_Operation(X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT]):
    """
    A base class for x86 operations that have one register that is read and written to.
    """

    register_in = operand_def(R1InvT)
    register_out = result_def(R1InvT)

    def __init__(
        self,
        register_in: SSAValue[R1InvT],
        *,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if register_out is None:
            register_out = register_in.type
        super().__init__(
            operands=[register_in],
            attributes={
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.register_in,)

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints((), (), ((self.register_in, self.register_out),))


class RM_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that have one register read and written to and one
    memory access with an optional offset.
    """

    register_in = operand_def(R1InvT)
    register_out = result_def(R1InvT)

    memory = operand_def(R2InvT)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        register_in: Operation | SSAValue,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        register_in = SSAValue.get(register_in)
        if register_out is None:
            register_out = cast(R1InvT, register_in.type)

        super().__init__(
            operands=[register_in, memory],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        destination = assembly_arg_str(self.register_in)
        return (destination, memory_access)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.memory,), (), ((self.register_in, self.register_out),)
        )


class DM_OperationHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.x86 import (
            DM_Operation_ConstantOffset,
        )

        return (DM_Operation_ConstantOffset(),)


class DM_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that load from memory into a destination register.
    """

    destination = result_def(R1InvT)
    memory = operand_def(R2InvT)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))

    traits = traits_def(
        DM_OperationHasCanonicalizationPatterns(),
        MemoryReadEffect(),
    )

    def __init__(
        self,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[memory],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        destination = assembly_arg_str(self.destination)
        return (destination, memory_access)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


class DI_Operation(X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT]):
    """
    A base class for x86 operations that have one destination register and an immediate
    value.
    """

    immediate = attr_def(IntegerAttr)
    destination = result_def(R1InvT)

    def __init__(
        self,
        immediate: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.destination, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        return {
            "immediate": parse_immediate_value(
                parser, IntegerType(32, Signedness.SIGNED)
            )
        }

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ", indent=0)
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class RI_Operation(X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT]):
    """
    A base class for x86 operations that have one register that is read and written to
    and an immediate value.
    """

    register_in = operand_def(R1InvT)
    register_out = result_def(R1InvT)

    immediate = attr_def(IntegerAttr)

    def __init__(
        self,
        register_in: Operation | SSAValue,
        immediate: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(comment, str):
            comment = StringAttr(comment)
        register_in = SSAValue.get(register_in)
        if register_out is None:
            register_out = cast(R1InvT, register_in.type)

        super().__init__(
            operands=[register_in],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.register_in, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(32, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints((), (), ((self.register_in, self.register_out),))


class MS_OperationHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.x86 import (
            MS_Operation_ConstantOffset,
        )

        return (MS_Operation_ConstantOffset(),)


class MS_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that have one memory reference and one source
    register.
    """

    memory = operand_def(R1InvT)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    source = operand_def(R2InvT)

    traits = traits_def(
        MS_OperationHasCanonicalizationPatterns(),
        MemoryReadEffect(),
        MemoryWriteEffect(),
    )

    def __init__(
        self,
        memory: Operation | SSAValue,
        source: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory, source],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return memory_access, self.source

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


class MI_Operation(X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT]):
    """
    A base class for x86 operations that have one memory reference and an immediate
    value.
    """

    memory = operand_def(R1InvT)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    immediate = attr_def(IntegerAttr)

    traits = traits_def(MemoryReadEffect(), MemoryWriteEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        immediate: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory],
            attributes={
                "immediate": immediate,
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        immediate = assembly_arg_str(self.immediate)
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return memory_access, immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(64, Signedness.SIGNED))
        attributes["immediate"] = temp
        if parser.parse_optional_punctuation(",") is not None:
            if offset := cls.parse_optional_memory_access_offset(parser):
                attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        if self.memory_offset.value.data != 0:
            printer.print_string(", ")
            print_immediate_value(printer, self.memory_offset)
        return {"immediate", "memory_offset"}


class DSI_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that have one destination register, one source
    register and an immediate value.
    """

    destination = result_def(R1InvT)
    source = operand_def(R2InvT)
    immediate = attr_def(IntegerAttr)

    def __init__(
        self,
        source: Operation | SSAValue,
        immediate: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.destination, self.source, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(32, Signedness.SIGNED))
        attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


class DMI_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT]
):
    """
    A base class for x86 operations that have one destination register, one memory
    reference and an immediate value.
    """

    destination = result_def(R1InvT)
    memory = operand_def(R2InvT)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    immediate = attr_def(IntegerAttr)
    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        immediate: int | IntegerAttr,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory],
            attributes={
                "immediate": immediate,
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        destination = assembly_arg_str(self.destination)
        immediate = assembly_arg_str(self.immediate)
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return destination, memory_access, immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(64, Signedness.SIGNED))
        attributes["immediate"] = temp
        if parser.parse_optional_punctuation(",") is not None:
            if offset := cls.parse_optional_memory_access_offset(parser):
                attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        if self.memory_offset.value.data != 0:
            printer.print_string(", ")
            print_immediate_value(printer, self.memory_offset)
        return {"immediate", "memory_offset"}


class M_Operation(X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT]):
    """
    A base class for x86 operations with a memory reference.
    """

    memory = operand_def(R1InvT)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    traits = traits_def(MemoryWriteEffect(), MemoryReadEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)

        super().__init__(
            operands=[memory],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


class ConditionalJumpOperation(X86Instruction, X86CustomFormatOperation, ABC):
    """
    A base class for Jcc operations.

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    rflags = operand_def(RFLAGS)

    then_values = var_operand_def(X86RegisterType)
    else_values = var_operand_def(X86RegisterType)

    irdl_options = [AttrSizedOperandSegments()]

    then_block = successor_def()
    else_block = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        rflags: Operation | SSAValue,
        then_values: Sequence[SSAValue],
        else_values: Sequence[SSAValue],
        then_block: Successor,
        else_block: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rflags, then_values, else_values],
            attributes={
                "comment": comment,
            },
            successors=(then_block, else_block),
        )

    def verify_(self) -> None:
        # The then block must start with a label op

        then_block_first_op = self.then_block.first_op

        if not isinstance(then_block_first_op, LabelOp):
            raise VerifyException("then block first op must be a label")

        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.then_values, self.then_block.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        for op_arg, block_arg in zip(self.else_values, self.else_block.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        # The else block must be the one immediately following this one

        parent_block = self.parent
        if parent_block is None:
            return

        parent_region = parent_block.parent
        if parent_region is None:
            return

        if parent_block.next_block is not self.else_block:
            raise VerifyException("else block must be immediately after op")

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        then_label = self.then_block.first_op
        assert isinstance(then_label, LabelOp)
        then_label_str = then_label.label.data
        if then_label_str.isdigit():
            # x86 Assembly: Numeric jump labels must be annotated with a suffix.
            # Jumping backward in code requires appending 'b' (e.g., "1b"), and
            # jumping forward requires appending 'f' (e.g., "1f").
            # Proper support for generating these labels is currently unimplemented.
            raise NotImplementedError(
                "Assembly printing for jumps to numeric labels not implemented"
            )
        return (then_label_str,)

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        print_type_pair(printer, self.rflags)
        printer.print_string(", ")
        printer.print_block_name(self.then_block)
        printer.print_string("(")
        printer.print_list(self.then_values, lambda val: print_type_pair(printer, val))
        printer.print_string("), ")
        printer.print_block_name(self.else_block)
        printer.print_string("(")
        printer.print_list(self.else_values, lambda val: print_type_pair(printer, val))
        printer.print_string(")")
        if self.attributes:
            printer.print_op_attributes(
                self.attributes,
                reserved_attr_names="operandSegmentSizes",
                print_keyword=True,
            )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        rflags = parse_type_pair(parser)
        parser.parse_punctuation(",")
        then_block = parser.parse_successor()
        then_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_type_pair(parser)
        )
        parser.parse_punctuation(",")
        else_block = parser.parse_successor()
        else_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_type_pair(parser)
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        op = cls(rflags, then_args, else_args, then_block, else_block)
        if attrs is not None:
            op.attributes |= attrs.data
        return op


class RSS_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT, R3InvT]
):
    """
    A base class for x86 operations that have one register that is read and written to,
    and two source registers.
    """

    register_in = operand_def(R1InvT)
    register_out = result_def(R1InvT)
    source1 = operand_def(R2InvT)
    source2 = operand_def(R3InvT)

    def __init__(
        self,
        register_in: SSAValue[R1InvT],
        source1: Operation | SSAValue,
        source2: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        if register_out is None:
            register_out = register_in.type

        super().__init__(
            operands=[register_in, source1, source2],
            attributes={
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.register_in, self.source1, self.source2

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1, self.source2), (), ((self.register_in, self.register_out),)
        )


class RSSK_Operation(X86Instruction, X86CustomFormatOperation, ABC):
    """
    A base class for x86 AVX512 operations that have one register r that is read and written to,
    and two source registers s1 and s2, with mask register k. The z attribute enables zero masking,
    which sets the elements of the destination register to zero where the corresponding
    bit in the mask is zero.
    """

    T: ClassVar[VarConstraint] = VarConstraint("T", base(AVX512RegisterType))

    register_in = operand_def(T)
    register_out = result_def(T)
    source1 = operand_def(AVX512RegisterType)
    source2 = operand_def(AVX512RegisterType)
    mask_reg = operand_def(AVX512MaskRegisterType)
    z = opt_attr_def(UnitAttr)

    def __init__(
        self,
        register_in: SSAValue[R1InvT],
        source1: Operation | SSAValue,
        source2: Operation | SSAValue,
        mask_reg: Operation | SSAValue,
        *,
        z: bool = False,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        if register_out is None:
            register_out = register_in.type

        super().__init__(
            operands=[register_in, source1, source2, mask_reg],
            attributes={
                "z": UnitAttr() if z else None,
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        register_in = masked_source_str(self.register_in, self.mask_reg, self.z)
        return register_in, self.source1, self.source2

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1, self.source2, self.mask_reg),
            (),
            ((self.register_in, self.register_out),),
        )


class DSS_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT, R3InvT]
):
    """
    A base class for x86 operations that have one destination register and two source
    registers.
    """

    destination = result_def(R1InvT)
    source1 = operand_def(R2InvT)
    source2 = operand_def(R3InvT)

    def __init__(
        self,
        source1: Operation | SSAValue[R2InvT],
        source2: Operation | SSAValue[R3InvT],
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source1, source2],
            attributes={
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.destination, self.source1, self.source2

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1, self.source2), (self.destination,), ()
        )


class RSM_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT, R4InvT]
):
    """
    A base class for x86 operations that have one register that is read and written to,
    one source register and one memory source operand.
    """

    register_in = operand_def(R1InvT)
    register_out = result_def(R1InvT)
    source1 = operand_def(R2InvT)
    memory = operand_def(R4InvT)
    memory_offset = attr_def(IntegerAttr[I32], default_value=IntegerAttr(0, 32))

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        register_in: SSAValue[R1InvT],
        source1: Operation | SSAValue,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr[I32],
        *,
        comment: str | StringAttr | None = None,
        register_out: R1InvT | None = None,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr[I32](memory_offset, 32)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        if register_out is None:
            register_out = register_in.type

        super().__init__(
            operands=[register_in, source1, memory],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[register_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        src1 = assembly_arg_str(self.source1)
        destination = assembly_arg_str(self.register_in)
        return destination, src1, memory_access

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser, i32):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}

    def get_register_constraints(self) -> RegisterConstraints:
        return RegisterConstraints(
            (self.source1, self.memory), (), ((self.register_in, self.register_out),)
        )


class DSSI_Operation(
    X86Instruction, X86CustomFormatOperation, ABC, Generic[R1InvT, R2InvT, R3InvT]
):
    """
    A base class for x86 operations that have one destination register, one source
    register and an immediate value.
    """

    destination = result_def(R1InvT)
    source0 = operand_def(R2InvT)
    source1 = operand_def(R3InvT)
    immediate = attr_def(IntegerAttr[IntegerType[8]])

    def __init__(
        self,
        source0: Operation | SSAValue,
        source1: Operation | SSAValue,
        immediate: int
        | IntegerAttr[IntegerType[Literal[8], Literal[Signedness.UNSIGNED]]],
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, IntegerType[8, Signedness.UNSIGNED](8, Signedness.UNSIGNED)
            )
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source0, source1],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.destination, self.source0, self.source1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(8, Signedness.UNSIGNED))
        attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


# endregion


class RS_AddOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.x86 import RS_Add_Zero

        return (RS_Add_Zero(),)


@irdl_op_definition
class RS_AddOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the registers r and s and stores the result in r.
    ```C
    x[r] = x[r] + x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/add).
    """

    name = "x86.rs.add"

    traits = traits_def(Pure(), RS_AddOpHasCanonicalizationPatterns())


@irdl_op_definition
class RS_SubOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    subtracts s from r and stores the result in r.
    ```C
    x[r] = x[r] - x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/sub).
    """

    name = "x86.rs.sub"


@irdl_op_definition
class RS_ImulOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the registers r and s and stores the result in r.
    ```C
    x[r] = x[r] * x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/imul).
    """

    name = "x86.rs.imul"


@irdl_op_definition
class RS_FAddOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the floating point values in registers r and s and stores the result in r.
    ```C
    x[r] += x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/fadd:faddp:fiadd).
    """

    name = "x86.rs.fadd"


@irdl_op_definition
class RS_FMulOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the floating point values in registers r and s and stores the result in
    r.
    ```C
    x[r] *= x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/fmul:fmulp:fimul).
    """

    name = "x86.rs.fmul"


@irdl_op_definition
class RS_AndOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r and s, stored in r
    ```C
    x[r] = x[r] & x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/and).
    """

    name = "x86.rs.and"


@irdl_op_definition
class RS_OrOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r and s, stored in r
    ```C
    x[r] = x[r] | x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/or).
    """

    name = "x86.rs.or"


@irdl_op_definition
class RS_XorOp(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r and s, stored in r
    ```C
    x[r] = x[r] ^ x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/xor).
    """

    name = "x86.rs.xor"


class DS_MovOpHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.x86 import RemoveRedundantDS_Mov

        return (RemoveRedundantDS_Mov(),)


@irdl_op_definition
class DS_MovOp(DS_Operation[X86RegisterType, GeneralRegisterType]):
    """
    Copies the value of s into r.
    ```C
    x[r] = x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/mov).
    """

    name = "x86.ds.mov"

    traits = traits_def(Pure(), DS_MovOpHasCanonicalizationPatterns())


@irdl_op_definition
class DS_VpbroadcastdOp(DS_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Broadcast single precision floating-point scalar in s to d.
    ```C
    x[r] = x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/vpbroadcast)
    """

    name = "x86.ds.vpbroadcastd"


@irdl_op_definition
class DS_VpbroadcastqOp(DS_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Broadcast double precision floating-point scalar in s to d.
    ```C
    x[r] = x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/vpbroadcast)
    """

    name = "x86.ds.vpbroadcastq"


@irdl_op_definition
class S_PushOp(X86Instruction, X86CustomFormatOperation):
    """
    Decreases %rsp and places s at the new memory location pointed to by %rsp.

    See external [documentation](https://www.felixcloutier.com/x86/push).
    """

    name = "x86.s.push"

    rsp_in = operand_def(RSP)
    rsp_out = result_def(RSP)
    source = operand_def(X86RegisterType)

    def __init__(
        self,
        rsp_in: Operation | SSAValue,
        source: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rsp_in, source],
            attributes={
                "comment": comment,
            },
            result_types=[RSP],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.source,)


@irdl_op_definition
class D_PopOp(X86Instruction, X86CustomFormatOperation):
    """
    Copies the value at the top of the stack into d and increases %rsp.

    See external [documentation](https://www.felixcloutier.com/x86/pop).
    """

    name = "x86.d.pop"

    rsp_in = operand_def(RSP)
    rsp_out = result_def(RSP)
    destination = result_def(X86RegisterType)

    def __init__(
        self,
        rsp_in: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        destination: X86RegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rsp_in],
            attributes={
                "comment": comment,
            },
            result_types=[RSP, destination],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.destination,)


@irdl_op_definition
class R_NegOp(R_Operation[GeneralRegisterType]):
    """
    Negates r and stores the result in r.
    ```C
    x[r] = -x[r]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/neg).
    """

    name = "x86.r.neg"


@irdl_op_definition
class R_NotOp(R_Operation[GeneralRegisterType]):
    """
    bitwise not of r, stored in r
    ```C
    x[r] = ~x[r]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/not).
    """

    name = "x86.r.not"


@irdl_op_definition
class R_IncOp(R_Operation[GeneralRegisterType]):
    """
    Increments r by 1 and stores the result in r.
    ```C
    x[r] = x[r] + 1
    ```

    See external [documentation](https://www.felixcloutier.com/x86/inc).
    """

    name = "x86.r.inc"


@irdl_op_definition
class R_DecOp(R_Operation[GeneralRegisterType]):
    """
    Decrements r by 1 and stores the result in r.
    ```C
    x[r] = x[r] - 1
    ```

    See external [documentation](https://www.felixcloutier.com/x86/dec).
    """

    name = "x86.r.dec"


@irdl_op_definition
class S_IDivOp(X86Instruction, X86CustomFormatOperation):
    """
    Divides the value in RDX:RAX by s and stores the quotient in RAX and the remainder
    in RDX.

    See external [documentation](https://www.felixcloutier.com/x86/idiv).
    """

    name = "x86.s.idiv"

    source = operand_def(X86RegisterType)
    rdx_input = operand_def(RDX)
    rax_input = operand_def(RAX)

    rdx_output = result_def(RDX)
    rax_output = result_def(RAX)

    def __init__(
        self,
        source: Operation | SSAValue,
        rdx_input: Operation | SSAValue,
        rax_input: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        rdx_output: GeneralRegisterType,
        rax_output: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source, rdx_input, rax_input],
            attributes={
                "comment": comment,
            },
            result_types=[rdx_output, rax_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.source,)


@irdl_op_definition
class S_ImulOp(X86Instruction, X86CustomFormatOperation):
    """
    The source operand is multiplied by the value in the RAX register and the product is
    stored in the RDX:RAX registers.
    ```C
    x[RDX:RAX] = x[RAX] * s
    ```

    See external [documentation](https://www.felixcloutier.com/x86/imul).
    """

    name = "x86.s.imul"

    source = operand_def(GeneralRegisterType)
    rax_input = operand_def(RAX)

    rdx_output = result_def(RDX)
    rax_output = result_def(RAX)

    def __init__(
        self,
        source: Operation | SSAValue,
        rax_input: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        rdx_output: GeneralRegisterType,
        rax_output: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source, rax_input],
            attributes={
                "comment": comment,
            },
            result_types=[rdx_output, rax_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.source,)


@irdl_op_definition
class RM_AddOp(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the value from the memory location pointed to by m to r and stores the result
    in r.
    ```C
    x[r] = x[r] + [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/add).
    """

    name = "x86.rm.add"


@irdl_op_definition
class RM_SubOp(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Subtracts the value from the memory location pointed to by m from r and stores the
    result in r.
    ```C
    x[r] = x[r] - [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/sub).
    """

    name = "x86.rm.sub"


@irdl_op_definition
class RM_ImulOp(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the value from the memory location pointed to by m with r and stores the
    result in r.
    ```C
    x[r] = x[r] * [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/imul).
    """

    name = "x86.rm.imul"


@irdl_op_definition
class RM_AndOp(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r and [m], stored in r
    ```C
    x[r] = x[r] & [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/and).
    """

    name = "x86.rm.and"


@irdl_op_definition
class RM_OrOp(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r and [m], stored in r
    ```C
    x[r] = x[r] | [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/or).
    """

    name = "x86.rm.or"


@irdl_op_definition
class RM_XorOp(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r and [m], stored in r
    ```C
    x[r] = x[r] ^ [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/xor).
    """

    name = "x86.rm.xor"


@irdl_op_definition
class DM_MovOp(DM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from the memory location pointed to by source register m into destination register d.
    ```C
    x[d] = [x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/mov).
    """

    name = "x86.dm.mov"


@irdl_op_definition
class DM_LeaOp(DM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Loads the effective address of the memory location pointed to by m into d.
    ```C
    x[d] = &x[m]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/lea).
    """

    name = "x86.dm.lea"


@irdl_op_definition
class RI_AddOp(RI_Operation[GeneralRegisterType]):
    """
    Adds the immediate value to r and stores the result in r.
    ```C
    x[r] = x[r] + immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/add).
    """

    name = "x86.ri.add"


@irdl_op_definition
class RI_SubOp(RI_Operation[GeneralRegisterType]):
    """
    Subtracts the immediate value from r and stores the result in r.
    ```C
    x[r] = x[r] - immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/sub).
    """

    name = "x86.ri.sub"


@irdl_op_definition
class RI_AndOp(RI_Operation[GeneralRegisterType]):
    """
    bitwise and of r and immediate, stored in r
    ```C
    x[r] = x[r] & immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/and).
    """

    name = "x86.ri.and"


@irdl_op_definition
class RI_OrOp(RI_Operation[GeneralRegisterType]):
    """
    bitwise or of r and immediate, stored in r
    ```C
    x[r] = x[r] | immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/or).
    """

    name = "x86.ri.or"


@irdl_op_definition
class RI_XorOp(RI_Operation[GeneralRegisterType]):
    """
    bitwise xor of r and immediate, stored in r
    ```C
    x[r] = x[r] ^ immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/xor).
    """

    name = "x86.ri.xor"


@irdl_op_definition
class DI_MovOp(DI_Operation[GeneralRegisterType]):
    """
    Copies the immediate value into r.
    ```C
    x[r] = immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/mov).
    """

    name = "x86.di.mov"

    traits = traits_def(Pure())


@irdl_op_definition
class MS_AddOp(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the value from s to the memory location pointed to by m.
    ```C
    [x[m]] = [x[m]] + x[s]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/add).
    """

    name = "x86.ms.add"


@irdl_op_definition
class MS_SubOp(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Subtracts the value from s from the memory location pointed to by m.
    [x[m]] = [x[m]] - x[s]

    See external [documentation](https://www.felixcloutier.com/x86/sub).
    """

    name = "x86.ms.sub"


@irdl_op_definition
class MS_AndOp(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of [m] and s
    [x[m]] = [x[m]] & x[s]

    See external [documentation](https://www.felixcloutier.com/x86/and).
    """

    name = "x86.ms.and"


@irdl_op_definition
class MS_OrOp(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of [m] and s
    [x[m]] = [x[m]] | x[s]

    See external [documentation](https://www.felixcloutier.com/x86/or).
    """

    name = "x86.ms.or"


@irdl_op_definition
class MS_XorOp(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of [m] and s
    [x[m]] = [x[m]] ^ x[s]

    See external [documentation](https://www.felixcloutier.com/x86/xor).
    """

    name = "x86.ms.xor"


@irdl_op_definition
class MS_MovOp(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from s into the memory location pointed to by m.
    [x[m]] = x[s]

    See external [documentation](https://www.felixcloutier.com/x86/mov).
    """

    name = "x86.ms.mov"


@irdl_op_definition
class MI_AddOp(MI_Operation[GeneralRegisterType]):
    """
    Adds the immediate value to the memory location pointed to by m.
    [x[m]] = [x[m]] + immediate

    See external [documentation](https://www.felixcloutier.com/x86/add).
    """

    name = "x86.mi.add"


@irdl_op_definition
class MI_SubOp(MI_Operation[GeneralRegisterType]):
    """
    Subtracts the immediate value from the memory location pointed to by m.
    [x[m]] = [x[m]] - immediate

    See external [documentation](https://www.felixcloutier.com/x86/sub).
    """

    name = "x86.mi.sub"


@irdl_op_definition
class MI_AndOp(MI_Operation[GeneralRegisterType]):
    """
    bitwise and of immediate and [m], stored in [m]
    ```C
    [x[m]] = [x[m]] & immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/and).
    """

    name = "x86.mi.and"


@irdl_op_definition
class MI_OrOp(MI_Operation[GeneralRegisterType]):
    """
    bitwise or of immediate and [m], stored in [m]
    ```C
    [x[m]] = [x[m]] | immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/or).
    """

    name = "x86.mi.or"


@irdl_op_definition
class MI_XorOp(MI_Operation[GeneralRegisterType]):
    """
    bitwise xor of immediate and [m], stored in [m]
    ```C
    [x[m]] = [x[m]] ^ immediate
    ```

    See external [documentation](https://www.felixcloutier.com/x86/xor).
    """

    name = "x86.mi.xor"


@irdl_op_definition
class MI_MovOp(MI_Operation[GeneralRegisterType]):
    """
    Copies the immediate value into the memory location pointed to by m.
    [x[m]] = immediate

    See external [documentation](https://www.felixcloutier.com/x86/mov).
    """

    name = "x86.mi.mov"


@irdl_op_definition
class DSI_ImulOp(DSI_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the immediate value with the source register and stores the result in the destination register.
    x[d] = x[s] * immediate

    See external [documentation](https://www.felixcloutier.com/x86/imul).
    """

    name = "x86.dsi.imul"


@irdl_op_definition
class DMI_ImulOp(DMI_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the immediate value with the memory location pointed to by m and stores the result in d.
    x[d] = [x[m]] * immediate

    See external [documentation](https://www.felixcloutier.com/x86/imul).
    """

    name = "x86.dmi.imul"


@irdl_op_definition
class M_PushOp(X86Instruction, X86CustomFormatOperation):
    """
    Decreases %rsp and places [m] at the new memory location pointed to by %rsp.

    See external [documentation](https://www.felixcloutier.com/x86/push).
    """

    name = "x86.m.push"

    rsp_in = operand_def(RSP)
    rsp_out = result_def(RSP)

    memory = operand_def(X86RegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))

    traits = traits_def(MemoryWriteEffect())

    def __init__(
        self,
        rsp_in: Operation | SSAValue,
        memory: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        memory_offset: int | IntegerAttr,
        rsp_out: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)

        super().__init__(
            operands=[rsp_in, memory],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[rsp_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


@irdl_op_definition
class M_PopOp(X86Instruction, X86CustomFormatOperation):
    """
    Copies the value at the top of the stack into [m] and increases %rsp.
    The value held by m is a pointer to the memory location where the value is stored.
    The only register modified by this operation is %rsp.

    See external [documentation](https://www.felixcloutier.com/x86/pop).
    """

    name = "x86.m.pop"

    rsp_in = operand_def(RSP)
    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    rsp_out = result_def(RSP)

    traits = traits_def(MemoryWriteEffect())

    def __init__(
        self,
        rsp_in: Operation | SSAValue,
        memory: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        memory_offset: int | IntegerAttr,
        rsp_out: GeneralRegisterType,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rsp_in, memory],
            attributes={
                "comment": comment,
            },
            result_types=[rsp_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["memory_offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


@irdl_op_definition
class M_NegOp(M_Operation[GeneralRegisterType]):
    """
    Negates the value at the memory location pointed to by m.
    ```C
    [x[m]] = -[x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/neg).
    """

    name = "x86.m.neg"


@irdl_op_definition
class M_NotOp(M_Operation[GeneralRegisterType]):
    """
    bitwise not of [m], stored in [m]
    ```C
    [x[m]] = ~[x[m]]
    ```

    See external [documentation](https://www.felixcloutier.com/x86/not).
    """

    name = "x86.m.not"


@irdl_op_definition
class M_IncOp(M_Operation[GeneralRegisterType]):
    """
    Increments the value at the memory location pointed to by m.
    [x[m]] = [x[m]] + 1

    See external [documentation](https://www.felixcloutier.com/x86/inc).
    """

    name = "x86.m.inc"


@irdl_op_definition
class M_DecOp(M_Operation[GeneralRegisterType]):
    """
    Decrements the value at the memory location pointed to by m.
    [x[m]] = [x[m]] - 1

    See external [documentation](https://www.felixcloutier.com/x86/dec).
    """

    name = "x86.m.dec"


@irdl_op_definition
class M_IDivOp(X86Instruction, X86CustomFormatOperation):
    """
    Divides the value in RDX:RAX by [m] and stores the quotient in RAX and the remainder in RDX.

    See external [documentation](https://www.felixcloutier.com/x86/idiv).
    """

    name = "x86.m.idiv"

    memory = operand_def(X86RegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    rdx_in = operand_def(RDX)
    rdx_out = result_def(RDX)
    rax_in = operand_def(RAX)
    rax_out = result_def(RAX)

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        rdx_in: Operation | SSAValue,
        rax_in: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        rdx_out: GeneralRegisterType,
        rax_out: GeneralRegisterType,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory, rdx_in, rax_in],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[rdx_out, rax_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        if offset := cls.parse_optional_memory_access_offset(parser):
            attributes["memory_offset"] = offset
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


@irdl_op_definition
class M_ImulOp(X86Instruction, X86CustomFormatOperation):
    """
    The source operand is multiplied by the value in the RAX register and the product is stored in the RDX:RAX registers.
    x[RDX:RAX] = x[RAX] * [x[m]]

    See external [documentation](https://www.felixcloutier.com/x86/imul).
    """

    name = "x86.m.imul"

    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))

    rdx_out = result_def(RDX)

    rax_in = operand_def(RAX)
    rax_out = result_def(RAX)

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        rax_in: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        rdx_out: GeneralRegisterType,
        rax_out: GeneralRegisterType,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory, rax_in],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[rdx_out, rax_out],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["memory_offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


@irdl_op_definition
class LabelOp(X86AsmOperation, X86CustomFormatOperation):
    """
    The label operation is used to emit text labels (e.g. loop:) that are used
    as branch, unconditional jump targets and symbol offsets.
    """

    name = "x86.label"
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
class DirectiveOp(X86AsmOperation, X86CustomFormatOperation):
    """
    The directive operation is used to represent a directive in the assembly code. (e.g. .globl; .type etc)
    """

    name = "x86.directive"

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
            },
        )

    def assembly_line(self) -> str | None:
        if self.value is not None and self.value.data:
            arg_str = assembly_arg_str(self.value.data)
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
class C_JmpOp(X86Instruction, X86CustomFormatOperation):
    """
    Unconditional jump to the label specified in destination.

    See external [documentation](https://www.felixcloutier.com/x86/jmp).
    """

    name = "x86.c.jmp"

    block_values = var_operand_def(X86RegisterType)

    successor = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        block_values: Sequence[SSAValue],
        successor: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[block_values],
            attributes={
                "comment": comment,
            },
            successors=(successor,),
        )

    def verify_(self) -> None:
        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.block_values, self.successor.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        if not isinstance(self.successor.first_op, LabelOp):
            raise VerifyException(
                "jmp operation successor must have a x86.label operation as a "
                f"first argument, found {self.successor.first_op}"
            )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_block_name(self.successor)
        printer.print_string("(")
        printer.print_list(self.block_values, lambda val: print_type_pair(printer, val))
        printer.print_string(")")
        if self.attributes:
            printer.print_op_attributes(self.attributes, print_keyword=True)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        successor = parser.parse_successor()
        block_values = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_type_pair(parser)
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        op = cls(block_values, successor)
        if attrs is not None:
            op.attributes |= attrs.data
        return op

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        dest_label = self.successor.first_op
        assert isinstance(dest_label, LabelOp)
        dest_label_str = dest_label.label.data
        if dest_label_str.isdigit():
            # x86 Assembly: Numeric jump labels must be annotated with a suffix.
            # Jumping backward in code requires appending 'b' (e.g., "1b"), and
            # jumping forward requires appending 'f' (e.g., "1f").
            # Proper support for generating these labels is currently unimplemented.
            raise NotImplementedError(
                "Assembly printing for jumps to numeric labels not implemented"
            )
        return (dest_label_str,)


@irdl_op_definition
class FallthroughOp(X86AsmOperation, X86CustomFormatOperation):
    """
    Continue execution into the next block.
    The successor of this operation must be immediately after this operation's parent.
    """

    name = "x86.fallthrough"

    block_values = var_operand_def(X86RegisterType)

    successor = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        block_values: Sequence[SSAValue],
        successor: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[block_values],
            attributes={
                "comment": comment,
            },
            successors=(successor,),
        )

    def verify_(self) -> None:
        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.block_values, self.successor.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        if (parent := self.parent) is not None:
            if parent.next_block is not self.successor:
                raise VerifyException(
                    "Fallthrough op successor must immediately follow its parent."
                )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_block_name(self.successor)
        printer.print_string("(")
        printer.print_list(self.block_values, lambda val: print_type_pair(printer, val))
        printer.print_string(")")
        if self.attributes:
            printer.print_op_attributes(self.attributes, print_keyword=True)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        successor = parser.parse_successor()
        block_values = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: parse_type_pair(parser)
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        op = cls(block_values, successor)
        if attrs is not None:
            op.attributes |= attrs.data
        return op

    def assembly_line(self) -> str | None:
        # Not printed in assembly
        return None


@irdl_op_definition
class SS_CmpOp(X86Instruction, X86CustomFormatOperation):
    """
    Compares the first source operand with the second source operand and sets the status
    flags in the EFLAGS register according to the results.

    See external [documentation](https://www.felixcloutier.com/x86/cmp).
    """

    name = "x86.ss.cmp"

    source1 = operand_def(X86RegisterType)
    source2 = operand_def(X86RegisterType)

    result = result_def(RFLAGSRegisterType)

    def __init__(
        self,
        source1: Operation | SSAValue,
        source2: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source1, source2],
            attributes={
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.source1, self.source2


@irdl_op_definition
class SM_CmpOp(X86Instruction, X86CustomFormatOperation):
    """
    Compares the first source operand with the second source operand and sets the status
    flags in the EFLAGS register according to the results.

    See external [documentation](https://www.felixcloutier.com/x86/cmp).
    """

    name = "x86.sm.cmp"

    source = operand_def(GeneralRegisterType)
    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))

    result = result_def(RFLAGSRegisterType)

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        source: Operation | SSAValue,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source, memory],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return self.source, memory_access

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["memory_offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


@irdl_op_definition
class SI_CmpOp(X86Instruction, X86CustomFormatOperation):
    """
    Compares the first source operand with the second source operand and sets the status
    flags in the EFLAGS register according to the results.

    See external [documentation](https://www.felixcloutier.com/x86/cmp).
    """

    name = "x86.si.cmp"

    source = operand_def(GeneralRegisterType)
    immediate = attr_def(IntegerAttr)

    result = result_def(RFLAGS)

    def __init__(
        self,
        source: Operation | SSAValue,
        immediate: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, 32)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[RFLAGS],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.source, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(32, Signedness.SIGNED))
        attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class MS_CmpOp(X86Instruction, X86CustomFormatOperation):
    """
    Compares the first source operand with the second source operand and sets the status
    flags in the EFLAGS register according to the results.

    See external [documentation](https://www.felixcloutier.com/x86/cmp).
    """

    name = "x86.ms.cmp"

    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    source = operand_def(GeneralRegisterType)

    result = result_def(RFLAGSRegisterType)

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        source: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory, source],
            attributes={
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return memory_access, self.source

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["memory_offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"memory_offset"}


@irdl_op_definition
class MI_CmpOp(X86Instruction, X86CustomFormatOperation):
    """
    Compares the first source operand with the second source operand and sets the status
    flags in the EFLAGS register according to the results.

    See external [documentation](https://www.felixcloutier.com/x86/cmp).
    """

    name = "x86.mi.cmp"

    memory = operand_def(GeneralRegisterType)
    memory_offset = attr_def(IntegerAttr, default_value=IntegerAttr(0, 64))
    immediate = attr_def(IntegerAttr)

    result = result_def(RFLAGSRegisterType)

    traits = traits_def(MemoryReadEffect())

    def __init__(
        self,
        memory: Operation | SSAValue,
        memory_offset: int | IntegerAttr,
        immediate: int | IntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(memory_offset, int):
            memory_offset = IntegerAttr(memory_offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[memory],
            attributes={
                "immediate": immediate,
                "memory_offset": memory_offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        immediate = assembly_arg_str(self.immediate)
        memory_access = memory_access_str(self.memory, self.memory_offset)
        return memory_access, immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(64, Signedness.SIGNED))
        attributes["immediate"] = temp
        if parser.parse_optional_punctuation(",") is not None:
            temp2 = parse_optional_immediate_value(
                parser, IntegerType(32, Signedness.SIGNED)
            )
            if temp2 is not None:
                attributes["memory_offset"] = temp2
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(", ")
        print_immediate_value(printer, self.immediate)
        printer.print_string(", ")
        print_immediate_value(printer, self.memory_offset)
        return {"immediate", "memory_offset"}


@irdl_op_definition
class C_JaOp(ConditionalJumpOperation):
    """
    Jump if above (CF=0 and ZF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.ja"


@irdl_op_definition
class C_JaeOp(ConditionalJumpOperation):
    """
    Jump if above or equal (CF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jae"


@irdl_op_definition
class C_JbOp(ConditionalJumpOperation):
    """
    Jump if below (CF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jb"


@irdl_op_definition
class C_JbeOp(ConditionalJumpOperation):
    """
    Jump if below or equal (CF=1 or ZF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jbe"


@irdl_op_definition
class C_JcOp(ConditionalJumpOperation):
    """
    Jump if carry (CF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jc"


@irdl_op_definition
class C_JeOp(ConditionalJumpOperation):
    """
    Jump if equal (ZF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.je"


@irdl_op_definition
class C_JgOp(ConditionalJumpOperation):
    """
    Jump if greater (ZF=0 and SF=OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jg"


@irdl_op_definition
class C_JgeOp(ConditionalJumpOperation):
    """
    Jump if greater or equal (SF=OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jge"


@irdl_op_definition
class C_JlOp(ConditionalJumpOperation):
    """
    Jump if less (SF≠OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jl"


@irdl_op_definition
class C_JleOp(ConditionalJumpOperation):
    """
    Jump if less or equal (ZF=1 or SF≠OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jle"


@irdl_op_definition
class C_JnaOp(ConditionalJumpOperation):
    """
    Jump if not above (CF=1 or ZF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jna"


@irdl_op_definition
class C_JnaeOp(ConditionalJumpOperation):
    """
    Jump if not above or equal (CF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnae"


@irdl_op_definition
class C_JnbOp(ConditionalJumpOperation):
    """
    Jump if not below (CF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnb"


@irdl_op_definition
class C_JnbeOp(ConditionalJumpOperation):
    """
    Jump if not below or equal (CF=0 and ZF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnbe"


@irdl_op_definition
class C_JncOp(ConditionalJumpOperation):
    """
    Jump if not carry (CF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnc"


@irdl_op_definition
class C_JneOp(ConditionalJumpOperation):
    """
    Jump if not equal (ZF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jne"


@irdl_op_definition
class C_JngOp(ConditionalJumpOperation):
    """
    Jump if not greater (ZF=1 or SF≠OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jng"


@irdl_op_definition
class C_JngeOp(ConditionalJumpOperation):
    """
    Jump if not greater or equal (SF≠OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnge"


@irdl_op_definition
class C_JnlOp(ConditionalJumpOperation):
    """
    Jump if not less (SF=OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnl"


@irdl_op_definition
class C_JnleOp(ConditionalJumpOperation):
    """
    Jump if not less or equal (ZF=0 and SF=OF).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnle"


@irdl_op_definition
class C_JnoOp(ConditionalJumpOperation):
    """
    Jump if not overflow (OF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jno"


@irdl_op_definition
class C_JnpOp(ConditionalJumpOperation):
    """
    Jump if not parity (PF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnp"


@irdl_op_definition
class C_JnsOp(ConditionalJumpOperation):
    """
    Jump if not sign (SF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jns"


@irdl_op_definition
class C_JnzOp(ConditionalJumpOperation):
    """
    Jump if not zero (ZF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jnz"


@irdl_op_definition
class C_JoOp(ConditionalJumpOperation):
    """
    Jump if overflow (OF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jo"


@irdl_op_definition
class C_JpOp(ConditionalJumpOperation):
    """
    Jump if parity (PF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jp"


@irdl_op_definition
class C_JpeOp(ConditionalJumpOperation):
    """
    Jump if parity even (PF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jpe"


@irdl_op_definition
class C_JpoOp(ConditionalJumpOperation):
    """
    Jump if parity odd (PF=0).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jpo"


@irdl_op_definition
class C_JsOp(ConditionalJumpOperation):
    """
    Jump if sign (SF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.js"


@irdl_op_definition
class C_JzOp(ConditionalJumpOperation):
    """
    Jump if zero (ZF=1).

    See external [documentation](https://www.felixcloutier.com/x86/jcc).
    """

    name = "x86.c.jz"


@irdl_op_definition
class RSS_Vfmadd231pdOp(
    RSS_Operation[X86VectorRegisterType, X86VectorRegisterType, X86VectorRegisterType]
):
    """
    Multiply packed double-precision floating-point elements in s1 and s2, add the
    intermediate result to r, and store the final result in r.

    See external [documentation](https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd).
    """

    name = "x86.rss.vfmadd231pd"


@irdl_op_definition
class RSSK_Vfmadd231pdOp(RSSK_Operation):
    """
    AVX512 masked multiply packed double-precision floating-point elements in s1 and s2, add the
    intermediate result to r, and store the final result in r.

    See external [documentation](https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd).
    """

    name = "x86.rssk.vfmadd231pd"


@irdl_op_definition
class RSM_Vfmadd231pdOp(
    RSM_Operation[X86VectorRegisterType, X86VectorRegisterType, GeneralRegisterType]
):
    """
    Multiply packed double-precision floating-point elements in s1 and at specified memory location, add the
    intermediate result to r, and store the final result in r.

    See external [documentation](https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd).
    """

    name = "x86.rsm.vfmadd231pd"


@irdl_op_definition
class RSS_Vfmadd231psOp(
    RSS_Operation[X86VectorRegisterType, X86VectorRegisterType, X86VectorRegisterType]
):
    """
    Multiply packed single-precision floating-point elements in s1 and s2, add the
    intermediate result to r, and store the final result in r.

    See external [documentation](https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd).
    """

    name = "x86.rss.vfmadd231ps"


@irdl_op_definition
class RSM_Vfmadd231psOp(
    RSM_Operation[X86VectorRegisterType, X86VectorRegisterType, GeneralRegisterType]
):
    """
    Multiply packed single-precision floating-point elements in s1 and at specified memory location, add the
    intermediate result to r, and store the final result in r.

    See external [documentation](https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd).
    """

    name = "x86.rsm.vfmadd231ps"


@irdl_op_definition
class DSS_AddpdOp(
    DSS_Operation[X86VectorRegisterType, X86VectorRegisterType, X86VectorRegisterType]
):
    """
    Add packed double-precision floating-point elements in s1 and s2 and store the
    result in d.

    See external [documentation](https://www.felixcloutier.com/x86/addpd).
    """

    name = "x86.dss.addpd"


@irdl_op_definition
class DSS_AddpsOp(
    DSS_Operation[X86VectorRegisterType, X86VectorRegisterType, X86VectorRegisterType]
):
    """
    Add packed single-precision floating-point elements in s1 and s2 and store the
    result in d.

    See external [documentation](https://www.felixcloutier.com/x86/addps).
    """

    name = "x86.dss.addps"


@irdl_op_definition
class DS_VmovapdOp(DS_Operation[X86VectorRegisterType, X86VectorRegisterType]):
    """
    Move aligned packed double precision floating-point values from zmm1 to zmm2

    See external [documentation](https://www.felixcloutier.com/x86/movapd).
    """

    name = "x86.ds.vmovapd"


@irdl_op_definition
class DSK_VmovapdOp(DSK_Operation):
    """
    Move aligned packed double precision floating-point values from zmm1 to zmm2 using
    writemask k1

    See external [documentation](https://www.felixcloutier.com/x86/movapd).
    """

    name = "x86.dsk.vmovapd"


@irdl_op_definition
class DS_VmovapsOp(DS_Operation[X86VectorRegisterType, X86VectorRegisterType]):
    """
    Move aligned packed single precision floating-point values from zmm1 to zmm2

    See external [documentation](https://www.felixcloutier.com/x86/movaps).
    """

    name = "x86.ds.vmovaps"


@irdl_op_definition
class MS_VmovapdOp(MS_Operation[GeneralRegisterType, X86VectorRegisterType]):
    """
    Move aligned packed double precision floating-point values from zmm1 to m512

    See external [documentation](https://www.felixcloutier.com/x86/movapd).
    """

    name = "x86.ms.vmovapd"


@irdl_op_definition
class MS_VmovapsOp(MS_Operation[GeneralRegisterType, X86VectorRegisterType]):
    """
    Move aligned packed single precision floating-point values from zmm1 to m512

    See external [documentation](https://www.felixcloutier.com/x86/movaps).
    """

    name = "x86.ms.vmovaps"


@irdl_op_definition
class MS_VmovupdOp(MS_Operation[GeneralRegisterType, X86VectorRegisterType]):
    """
    Move unaligned packed double precision floating-point values from vector register to memory

    See external [documentation](https://www.felixcloutier.com/x86/movupd).
    """

    name = "x86.ms.vmovupd"


@irdl_op_definition
class MS_VmovupsOp(MS_Operation[GeneralRegisterType, X86VectorRegisterType]):
    """
    Move unaligned packed single precision floating-point values from vector register to memory

    See external [documentation](https://www.felixcloutier.com/x86/movups).
    """

    name = "x86.ms.vmovups"


@irdl_op_definition
class DM_VmovapdOp(DM_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Move aligned packed double precision floating-point values from memory to vector
    register.

    See external [documentation](https://www.felixcloutier.com/x86/movapd).
    """

    name = "x86.dm.vmovapd"


@irdl_op_definition
class DM_VmovapsOp(DM_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Move aligned packed single precision floating-point values from memory to vector
    register.

    See external [documentation](https://www.felixcloutier.com/x86/movaps).
    """

    name = "x86.dm.vmovaps"


@irdl_op_definition
class DM_VmovupdOp(DM_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Move unaligned packed double precision floating-point values from memory to vector
    register.

    See external [documentation](https://www.felixcloutier.com/x86/movupd).
    """

    name = "x86.dm.vmovupd"


@irdl_op_definition
class DM_VmovupsOp(DM_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Move unaligned packed single precision floating-point values from memory to vector
    register.

    See external [documentation](https://www.felixcloutier.com/x86/movups).
    """

    name = "x86.dm.vmovups"


@irdl_op_definition
class MS_VmovntpdOp(MS_Operation[GeneralRegisterType, X86VectorRegisterType]):
    """
    Moves the packed double precision floating-point values in the source operand to the
    destination operand using a non-temporal hint to prevent caching of the data during
    the write to memory.

    See external [documentation](https://www.felixcloutier.com/x86/movntpd).
    """

    name = "x86.ms.vmovntpd"


@irdl_op_definition
class MS_VmovntpsOp(MS_Operation[GeneralRegisterType, X86VectorRegisterType]):
    """
    Moves the packed single precision floating-point values in the source operand to the
    destination operand using a non-temporal hint to prevent caching of the data during
    the write to memory.

    See external [documentation](https://www.felixcloutier.com/x86/movntps).
    """

    name = "x86.ms.vmovntps"


@irdl_op_definition
class DM_VbroadcastsdOp(DM_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Broadcast low double precision floating-point element in m64 to eight locations in zmm1 using writemask k1

    See external [documentation](https://www.felixcloutier.com/x86/vbroadcast).
    """

    name = "x86.dm.vbroadcastsd"


@irdl_op_definition
class DM_VbroadcastssOp(DM_Operation[X86VectorRegisterType, GeneralRegisterType]):
    """
    Broadcast single precision floating-point element to eight locations in memory

    See external [documentation](https://www.felixcloutier.com/x86/vbroadcast).
    """

    name = "x86.dm.vbroadcastss"


@irdl_op_definition
class DSSI_ShufpsOp(
    DSSI_Operation[X86VectorRegisterType, X86VectorRegisterType, X86VectorRegisterType]
):
    """
    Selects a single precision floating-point value of an input quadruplet using a
    two-bit control and move to a designated element of the destination operand.
    Each 64-bit element-pair of a 128-bit lane of the destination operand is interleaved
    between the corresponding lane of the first source operand and the second source
    operand at the granularity 128 bits. Each two bits in the imm8 byte, starting from
    bit 0, is the select control of the corresponding element of a 128-bit lane of the
    destination to received the shuffled result of an input quadruplet. The two lower
    elements of a 128-bit lane in the destination receives shuffle results from the
    quadruple of the first source operand. The next two elements of the destination
    receives shuffle results from the quadruple of the second source operand.

    See external [documentation](https://www.felixcloutier.com/x86/shufps)
    """

    name = "x86.dssi.shufps"


class GetAnyRegisterOperation(
    X86AsmOperation, X86CustomFormatOperation, ABC, Generic[R1InvT]
):
    """
    This instruction allows us to create an SSAValue for a given register name.
    """

    result: OpResult[R1InvT] = result_def(R1InvT)

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
class GetAVXRegisterOp(GetAnyRegisterOperation[X86VectorRegisterType]):
    name = "x86.get_avx_register"


@irdl_op_definition
class GetMaskRegisterOp(GetAnyRegisterOperation[AVX512MaskRegisterType]):
    name = "x86.get_mask_register"


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    printer = AssemblyPrinter(stream=output)
    print(".intel_syntax noprefix", file=output)
    printer.print_module(module)


def x86_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()
