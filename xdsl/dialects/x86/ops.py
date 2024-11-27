from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from io import StringIO
from typing import IO, Generic, TypeVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    StringAttr,
)
from xdsl.dialects.func import FuncOp
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Successor,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

from .assembly import (
    AssemblyInstructionArg,
    append_comment,
    assembly_arg_str,
    assembly_line,
    memory_access_str,
    parse_immediate_value,
    parse_optional_immediate_value,
    parse_type_pair,
    print_immediate_value,
    print_type_pair,
)
from .attributes import LabelAttr
from .register import (
    AVXRegisterType,
    GeneralRegisterType,
    RFLAGSRegisterType,
    X86RegisterType,
)

R1InvT = TypeVar("R1InvT", bound=X86RegisterType)
R2InvT = TypeVar("R2InvT", bound=X86RegisterType)
R3InvT = TypeVar("R3InvT", bound=X86RegisterType)


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


class X86Instruction(X86Op):
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

        return Dialect.split_name(self.name)[1]

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            assembly_arg_str(arg)
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return assembly_line(instruction_name, arg_str, self.comment)


class R_RR_Operation(Generic[R1InvT, R2InvT], IRDLOperation, X86Instruction, ABC):
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


@irdl_op_definition
class RR_AddOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the registers r1 and r2 and stores the result in r1.
    x[r1] = x[r1] + x[r2]
    https://www.felixcloutier.com/x86/add
    """

    name = "x86.rr.add"


@irdl_op_definition
class RR_SubOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    subtracts r2 from r1 and stores the result in r1.
    x[r1] = x[r1] - x[r2]
    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.rr.sub"


@irdl_op_definition
class RR_ImulOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the registers r1 and r2 and stores the result in r1.
    x[r1] = x[r1] * x[r2]
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.rr.imul"


@irdl_op_definition
class RR_AndOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r1 and r2, stored in r1
    x[r1] = x[r1] & x[r2]
    https://www.felixcloutier.com/x86/and
    """

    name = "x86.rr.and"


@irdl_op_definition
class RR_OrOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r1 and r2, stored in r1
    x[r1] = x[r1] | x[r2]
    https://www.felixcloutier.com/x86/or
    """

    name = "x86.rr.or"


@irdl_op_definition
class RR_XorOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r1 and r2, stored in r1
    x[r1] = x[r1] ^ x[r2]
    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.rr.xor"


@irdl_op_definition
class RR_MovOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value of r1 into r2.
    x[r1] = x[r2]
    https://www.felixcloutier..com/x86/mov
    """

    name = "x86.rr.mov"


@irdl_op_definition
class R_PushOp(IRDLOperation, X86Instruction):
    """
    Decreases %rsp and places r1 at the new memory location pointed to by %rsp.
    https://www.felixcloutier.com/x86/push
    """

    name = "x86.r.push"

    rsp_input = operand_def(GeneralRegisterType("rsp"))
    source = operand_def(R1InvT)
    rsp_output = result_def(GeneralRegisterType("rsp"))

    def __init__(
        self,
        rsp_input: Operation | SSAValue,
        source: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        rsp_output: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rsp_input, source],
            attributes={
                "comment": comment,
            },
            result_types=[rsp_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.source,)


@irdl_op_definition
class R_PopOp(IRDLOperation, X86Instruction):
    """
    Copies the value at the top of the stack into r1 and increases %rsp.
    https://www.felixcloutier.com/x86/pop
    """

    name = "x86.r.pop"

    rsp_input = operand_def(GeneralRegisterType("rsp"))
    destination = result_def(R1InvT)
    rsp_output = result_def(GeneralRegisterType("rsp"))

    def __init__(
        self,
        rsp_input: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        destination: X86RegisterType,
        rsp_output: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rsp_input],
            attributes={
                "comment": comment,
            },
            result_types=[destination, rsp_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.destination,)


class R_R_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one register acting as both source and destination.
    """

    source = operand_def(R1InvT)
    destination = result_def(R1InvT)

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
        return (self.source,)


@irdl_op_definition
class R_NegOp(R_R_Operation[GeneralRegisterType]):
    """
    Negates r1 and stores the result in r1.
    x[r1] = -x[r1]
    https://www.felixcloutier.com/x86/neg
    """

    name = "x86.r.neg"


@irdl_op_definition
class R_NotOp(R_R_Operation[GeneralRegisterType]):
    """
    bitwise not of r1, stored in r1
    x[r1] = ~x[r1]
    https://www.felixcloutier.com/x86/not
    """

    name = "x86.r.not"


@irdl_op_definition
class R_IncOp(R_R_Operation[GeneralRegisterType]):
    """
    Increments r1 by 1 and stores the result in r1.
    x[r1] = x[r1] + 1
    https://www.felixcloutier.com/x86/inc
    """

    name = "x86.r.inc"


@irdl_op_definition
class R_DecOp(R_R_Operation[GeneralRegisterType]):
    """
    Decrements r1 by 1 and stores the result in r1.
    x[r1] = x[r1] - 1
    https://www.felixcloutier.com/x86/dec
    """

    name = "x86.r.dec"


@irdl_op_definition
class R_IDivOp(IRDLOperation, X86Instruction):
    """
    Divides the value in RDX:RAX by r1 and stores the quotient in RAX and the remainder in RDX.
    https://www.felixcloutier.com/x86/idiv
    """

    name = "x86.r.idiv"

    r1 = operand_def(R1InvT)
    rdx_input = operand_def(GeneralRegisterType("rdx"))
    rax_input = operand_def(GeneralRegisterType("rax"))

    rdx_output = result_def(GeneralRegisterType("rdx"))
    rax_output = result_def(GeneralRegisterType("rax"))

    def __init__(
        self,
        r1: Operation | SSAValue,
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
            operands=[r1, rdx_input, rax_input],
            attributes={
                "comment": comment,
            },
            result_types=[rdx_output, rax_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.r1,)


@irdl_op_definition
class R_ImulOp(IRDLOperation, X86Instruction):
    """
    The source operand is multiplied by the value in the RAX register and the product is stored in the RDX:RAX registers.
    x[RDX:RAX] = x[RAX] * r1
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.r.imul"

    r1 = operand_def(GeneralRegisterType)
    rax_input = operand_def(GeneralRegisterType("rax"))

    rdx_output = result_def(GeneralRegisterType("rdx"))
    rax_output = result_def(GeneralRegisterType("rax"))

    def __init__(
        self,
        r1: Operation | SSAValue,
        rax_input: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        rdx_output: GeneralRegisterType,
        rax_output: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, rax_input],
            attributes={
                "comment": comment,
            },
            result_types=[rdx_output, rax_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.r1,)


class R_RM_Operation(Generic[R1InvT, R2InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one register and one memory access with an optional offset.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    offset = opt_attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None = None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
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
        memory_access = memory_access_str(self.r2, self.offset)
        destination = assembly_arg_str(self.r1)
        return (destination, memory_access)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class RM_AddOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the value from the memory location pointed to by r2 to r1 and stores the result in r1.
    x[r1] = x[r1] + [x[r2]]
    https://www.felixcloutier.com/x86/add
    """

    name = "x86.rm.add"


@irdl_op_definition
class RM_SubOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Subtracts the value from the memory location pointed to by r2 from r1 and stores the result in r1.
    x[r1] = x[r1] - [x[r2]]
    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.rm.sub"


@irdl_op_definition
class RM_ImulOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the value from the memory location pointed to by r2 with r1 and stores the result in r1.
    x[r1] = x[r1] * [x[r2]]
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.rm.imul"


@irdl_op_definition
class RM_AndOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r1 and [r2], stored in r1
    x[r1] = x[r1] & [x[r2]]
    https://www.felixcloutier.com/x86/and
    """

    name = "x86.rm.and"


@irdl_op_definition
class RM_OrOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r1 and [r2], stored in r1
    x[r1] = x[r1] | [x[r2]]
    https://www.felixcloutier.com/x86/or
    """

    name = "x86.rm.or"


@irdl_op_definition
class RM_XorOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r1 and [r2], stored in r1
    x[r1] = x[r1] ^ [x[r2]]
    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.rm.xor"


@irdl_op_definition
class RM_MovOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from the memory location pointed to by r2 into r1.
    x[r1] = [x[r2]]
    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.rm.mov"


@irdl_op_definition
class RM_leaOp(R_RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Loads the effective address of the memory location pointed to by r2 into r1.
    x[r1] = &x[r2]
    https://www.felixcloutier.com/x86/lea
    """

    name = "x86.rm.lea"


class R_RI_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one register and an immediate value.
    """

    r1 = operand_def(R1InvT)
    immediate = attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the deault immediate size is 32 bits
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(32, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class RI_AddOp(R_RI_Operation[GeneralRegisterType]):
    """
    Adds the immediate value to r1 and stores the result in r1.
    x[r1] = x[r1] + immediate
    https://www.felixcloutier.com/x86/add
    """

    name = "x86.ri.add"


@irdl_op_definition
class RI_SubOp(R_RI_Operation[GeneralRegisterType]):
    """
    Subtracts the immediate value from r1 and stores the result in r1.
    x[r1] = x[r1] - immediate
    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.ri.sub"


@irdl_op_definition
class RI_AndOp(R_RI_Operation[GeneralRegisterType]):
    """
    bitwise and of r1 and immediate, stored in r1
    x[r1] = x[r1] & immediate
    https://www.felixcloutier.com/x86/and
    """

    name = "x86.ri.and"


@irdl_op_definition
class RI_OrOp(R_RI_Operation[GeneralRegisterType]):
    """
    bitwise or of r1 and immediate, stored in r1
    x[r1] = x[r1] | immediate
    https://www.felixcloutier.com/x86/or
    """

    name = "x86.ri.or"


@irdl_op_definition
class RI_XorOp(R_RI_Operation[GeneralRegisterType]):
    """
    bitwise xor of r1 and immediate, stored in r1
    x[r1] = x[r1] ^ immediate
    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.ri.xor"


@irdl_op_definition
class RI_MovOp(R_RI_Operation[GeneralRegisterType]):
    """
    Copies the immediate value into r1.
    x[r1] = immediate
    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.ri.mov"


class M_MR_Operation(Generic[R1InvT, R2InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one memory reference and one register.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    offset = opt_attr_def(AnyIntegerAttr)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.r1, self.offset)
        return memory_access, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class MR_AddOp(M_MR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the value from r2 to the memory location pointed to by r1.
    [x[r1]] = [x[r1]] + x[r2]
    https://www.felixcloutier.com/x86/add
    """

    name = "x86.mr.add"


@irdl_op_definition
class MR_SubOp(M_MR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Subtracts the value from r2 from the memory location pointed to by r1.
    [x[r1]] = [x[r1]] - x[r2]
    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.mr.sub"


@irdl_op_definition
class MR_AndOp(M_MR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of [r1] and r2
    [x[r1]] = [x[r1]] & x[r2]
    https://www.felixcloutier.com/x86/and
    """

    name = "x86.mr.and"


@irdl_op_definition
class MR_OrOp(M_MR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of [r1] and r2
    [x[r1]] = [x[r1]] | x[r2]
    https://www.felixcloutier.com/x86/or
    """

    name = "x86.mr.or"


@irdl_op_definition
class MR_XorOp(M_MR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of [r1] and r2
    [x[r1]] = [x[r1]] ^ x[r2]
    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.mr.xor"


@irdl_op_definition
class MR_MovOp(M_MR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from r2 into the memory location pointed to by r1.
    [x[r1]] = x[r2]
    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.mr.mov"


class M_MI_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one memory reference and an immediate value.
    """

    r1 = operand_def(R1InvT)
    immediate = attr_def(AnyIntegerAttr)
    offset = opt_attr_def(AnyIntegerAttr)

    def __init__(
        self,
        r1: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the deault immediate size is 32 bits
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "immediate": immediate,
                "offset": offset,
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        immediate = assembly_arg_str(self.immediate)
        memory_access = memory_access_str(self.r1, self.offset)
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
                attributes["offset"] = temp2
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        print_immediate_value(printer, self.immediate)
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"immediate", "offset"}


@irdl_op_definition
class MI_AddOp(M_MI_Operation[GeneralRegisterType]):
    """
    Adds the immediate value to the memory location pointed to by r1.
    [x[r1]] = [x[r1]] + immediate
    https://www.felixcloutier.com/x86/add
    """

    name = "x86.mi.add"


@irdl_op_definition
class MI_SubOp(M_MI_Operation[GeneralRegisterType]):
    """
    Subtracts the immediate value from the memory location pointed to by r1.
    [x[r1]] = [x[r1]] - immediate
    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.mi.sub"


@irdl_op_definition
class MI_AndOp(M_MI_Operation[GeneralRegisterType]):
    """
    bitwise and of immediate and [r1], stored in [r1]
    [x[r1]] = [x[r1]] & immediate
    https://www.felixcloutier.com/x86/and
    """

    name = "x86.mi.and"


@irdl_op_definition
class MI_OrOp(M_MI_Operation[GeneralRegisterType]):
    """
    bitwise or of immediate and [r1], stored in [r1]
    [x[r1]] = [x[r1]] | immediate
    https://www.felixcloutier.com/x86/or
    """

    name = "x86.mi.or"


@irdl_op_definition
class MI_XorOp(M_MI_Operation[GeneralRegisterType]):
    """
    bitwise xor of immediate and [r1], stored in [r1]
    [x[r1]] = [x[r1]] ^ immediate
    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.mi.xor"


@irdl_op_definition
class MI_MovOp(M_MI_Operation[GeneralRegisterType]):
    """
    Copies the immediate value into the memory location pointed to by r1.
    [x[r1]] = immediate
    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.mi.mov"


class R_RRI_Operation(Generic[R1InvT, R2InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one destination register, one source register and an immediate value.
    """

    r2 = operand_def(R2InvT)
    immediate = attr_def(AnyIntegerAttr)

    r1 = result_def(R1InvT)

    def __init__(
        self,
        r2: Operation | SSAValue,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        r1: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r2],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[r1],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(32, Signedness.SIGNED))
        attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class RRI_ImulOp(R_RRI_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the immediate value with the source register and stores the result in the destination register.
    x[r1] = x[r2] * immediate
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.rri.imul"


class R_RMI_Operation(Generic[R1InvT, R2InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one source register, one memory reference and an immediate value.
    """

    r2 = operand_def(R2InvT)
    immediate = attr_def(AnyIntegerAttr)
    offset = opt_attr_def(AnyIntegerAttr)

    r1 = result_def(R1InvT)

    def __init__(
        self,
        r2: Operation | SSAValue,
        immediate: int | AnyIntegerAttr,
        offset: int | AnyIntegerAttr | None,
        *,
        comment: str | StringAttr | None = None,
        r1: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the default immediate size is 32 bits
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r2],
            attributes={
                "immediate": immediate,
                "offset": offset,
                "comment": comment,
            },
            result_types=[r1],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        destination = assembly_arg_str(self.r1)
        immediate = assembly_arg_str(self.immediate)
        memory_access = memory_access_str(self.r2, self.offset)
        return destination, memory_access, immediate

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
                attributes["offset"] = temp2
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        print_immediate_value(printer, self.immediate)
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"immediate", "offset"}


@irdl_op_definition
class RMI_ImulOp(R_RMI_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the immediate value with the memory location pointed to by r2 and stores the result in r1.
    x[r1] = [x[r2]] * immediate
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.rmi.imul"


@irdl_op_definition
class M_PushOp(IRDLOperation, X86Instruction):
    """
    Decreases %rsp and places [r1] at the new memory location pointed to by %rsp.
    https://www.felixcloutier.com/x86/push
    """

    name = "x86.m.push"

    rsp_input = operand_def(GeneralRegisterType("rsp"))
    source = operand_def(R1InvT)
    offset = opt_attr_def(AnyIntegerAttr)
    rsp_output = result_def(GeneralRegisterType("rsp"))

    def __init__(
        self,
        rsp_input: Operation | SSAValue,
        source: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        offset: int | AnyIntegerAttr | None,
        rsp_output: GeneralRegisterType,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)

        super().__init__(
            operands=[rsp_input, source],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[rsp_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.source, self.offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class M_PopOp(IRDLOperation, X86Instruction):
    """
    Copies the value at the top of the stack into [r1] and increases %rsp.
    The value held by r1 is a pointer to the memory location where the value is stored.
    The only register modified by this operation is %rsp.
    https://www.felixcloutier.com/x86/pop
    """

    name = "x86.m.pop"

    rsp_input = operand_def(GeneralRegisterType("rsp"))
    destination = operand_def(
        GeneralRegisterType
    )  # the destination is a pointer to the memory location and the register itself is not modified
    offset = opt_attr_def(AnyIntegerAttr)
    rsp_output = result_def(GeneralRegisterType("rsp"))

    def __init__(
        self,
        rsp_input: Operation | SSAValue,
        destination: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        offset: int | AnyIntegerAttr | None = None,
        rsp_output: GeneralRegisterType,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rsp_input, destination],
            attributes={
                "comment": comment,
            },
            result_types=[rsp_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.destination, self.offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"offset"}


class M_M_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations with a memory reference that's both a source and a destination
    """

    source = operand_def(R1InvT)
    offset = opt_attr_def(AnyIntegerAttr)

    def __init__(
        self,
        source: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)

        super().__init__(
            operands=[source],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.source, self.offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class M_NegOp(M_M_Operation[GeneralRegisterType]):
    """
    Negates the value at the memory location pointed to by r1.
    [x[r1]] = -[x[r1]]
    https://www.felixcloutier.com/x86/neg
    """

    name = "x86.m.neg"


@irdl_op_definition
class M_NotOp(M_M_Operation[GeneralRegisterType]):
    """
    bitwise not of [r1], stored in [r1]
    [x[r1]] = ~[x[r1]]
    https://www.felixcloutier.com/x86/not
    """

    name = "x86.m.not"


@irdl_op_definition
class M_IncOp(M_M_Operation[GeneralRegisterType]):
    """
    Increments the value at the memory location pointed to by r1.
    [x[r1]] = [x[r1]] + 1
    https://www.felixcloutier.com/x86/inc
    """

    name = "x86.m.inc"


@irdl_op_definition
class M_DecOp(M_M_Operation[GeneralRegisterType]):
    """
    Decrements the value at the memory location pointed to by r1.
    [x[r1]] = [x[r1]] - 1
    https://www.felixcloutier.com/x86/dec
    """

    name = "x86.m.dec"


@irdl_op_definition
class M_IDivOp(IRDLOperation, X86Instruction):
    """
    Divides the value in RDX:RAX by [r1] and stores the quotient in RAX and the remainder in RDX.
    https://www.felixcloutier.com/x86/idiv
    """

    name = "x86.m.idiv"

    r1 = operand_def(R1InvT)
    rdx_input = operand_def(GeneralRegisterType("rdx"))
    rax_input = operand_def(GeneralRegisterType("rax"))
    offset = opt_attr_def(AnyIntegerAttr)

    rdx_output = result_def(GeneralRegisterType("rdx"))
    rax_output = result_def(GeneralRegisterType("rax"))

    def __init__(
        self,
        r1: Operation | SSAValue,
        rdx_input: Operation | SSAValue,
        rax_input: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None = None,
        *,
        comment: str | StringAttr | None = None,
        rdx_output: GeneralRegisterType,
        rax_output: GeneralRegisterType,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, rdx_input, rax_input],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[rdx_output, rax_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.r1, self.offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class M_ImulOp(IRDLOperation, X86Instruction):
    """
    The source operand is multiplied by the value in the RAX register and the product is stored in the RDX:RAX registers.
    x[RDX:RAX] = x[RAX] * [x[r1]]
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.m.imul"

    r1 = operand_def(GeneralRegisterType)
    rax_input = operand_def(GeneralRegisterType("rax"))
    offset = opt_attr_def(AnyIntegerAttr)

    rdx_output = result_def(GeneralRegisterType("rdx"))
    rax_output = result_def(GeneralRegisterType("rax"))

    def __init__(
        self,
        r1: Operation | SSAValue,
        rax_input: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None = None,
        *,
        comment: str | StringAttr | None = None,
        rdx_output: GeneralRegisterType,
        rax_output: GeneralRegisterType,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, rax_input],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[rdx_output, rax_output],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        memory_access = memory_access_str(self.r1, self.offset)
        return (memory_access,)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class LabelOp(IRDLOperation, X86Op):
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
        return append_comment(f"{self.label.data}:", self.comment)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["label"] = LabelAttr(parser.parse_str_literal("Expected label"))
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(" ")
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
class DirectiveOp(IRDLOperation, X86Op):
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

        return assembly_line(self.directive.data, arg_str, is_indented=False)

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
class S_JmpOp(IRDLOperation, X86Instruction):
    """
    Unconditional jump to the label specified in destination.
    https://www.felixcloutier.com/x86/jmp
    """

    name = "x86.s.jmp"

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
        return (dest_label.label,)


@irdl_op_definition
class RR_CmpOp(IRDLOperation, X86Instruction):
    """
    Compares the first source operand with the second source operand and sets the status flags in the EFLAGS register according to the results.
    https://www.felixcloutier.com/x86/cmp
    """

    name = "x86.rr.cmp"

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)

    result = result_def(RFLAGSRegisterType)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
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


@irdl_op_definition
class RM_CmpOp(IRDLOperation, X86Instruction):
    """
    Compares the first source operand with the second source operand and sets the status flags in the EFLAGS register according to the results.
    https://www.felixcloutier.com/x86/cmp
    """

    name = "x86.rm.cmp"

    r1 = operand_def(GeneralRegisterType)
    r2 = operand_def(GeneralRegisterType)
    offset = opt_attr_def(AnyIntegerAttr)

    result = result_def(RFLAGSRegisterType)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
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
        memory_access = memory_access_str(self.r2, self.offset)
        return self.r1, memory_access

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class RI_CmpOp(IRDLOperation, X86Instruction):
    """
    Compares the first source operand with the second source operand and sets the status flags in the EFLAGS register according to the results.
    https://www.felixcloutier.com/x86/cmp
    """

    name = "x86.ri.cmp"

    r1 = operand_def(GeneralRegisterType)
    immediate = attr_def(AnyIntegerAttr)

    result = result_def(RFLAGSRegisterType)

    def __init__(
        self,
        r1: Operation | SSAValue,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, 32)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.r1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_immediate_value(parser, IntegerType(32, Signedness.SIGNED))
        attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        print_immediate_value(printer, self.immediate)
        return {"immediate"}


@irdl_op_definition
class MR_CmpOp(IRDLOperation, X86Instruction):
    """
    Compares the first source operand with the second source operand and sets the status flags in the EFLAGS register according to the results.
    https://www.felixcloutier.com/x86/cmp
    """

    name = "x86.mr.cmp"

    r1 = operand_def(GeneralRegisterType)
    r2 = operand_def(GeneralRegisterType)
    offset = opt_attr_def(AnyIntegerAttr)

    result = result_def(RFLAGSRegisterType)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
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
        memory_access = memory_access_str(self.r1, self.offset)
        return memory_access, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"offset"}


@irdl_op_definition
class MI_CmpOp(IRDLOperation, X86Instruction):
    """
    Compares the first source operand with the second source operand and sets the status flags in the EFLAGS register according to the results.
    https://www.felixcloutier.com/x86/cmp
    """

    name = "x86.mi.cmp"

    r1 = operand_def(GeneralRegisterType)
    immediate = attr_def(AnyIntegerAttr)
    offset = opt_attr_def(AnyIntegerAttr)

    result = result_def(RFLAGSRegisterType)

    def __init__(
        self,
        r1: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: RFLAGSRegisterType,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(
                immediate, 32
            )  # the deault immediate size is 32 bits
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 64)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "immediate": immediate,
                "offset": offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        immediate = assembly_arg_str(self.immediate)
        memory_access = memory_access_str(self.r1, self.offset)
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
                attributes["offset"] = temp2
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        print_immediate_value(printer, self.immediate)
        if self.offset is not None:
            printer.print(", ")
            print_immediate_value(printer, self.offset)
        return {"immediate", "offset"}


class ConditionalJumpOperation(IRDLOperation, X86Instruction, ABC):
    """
    A base class for Jcc operations.
    https://www.felixcloutier.com/x86/jcc
    """

    rflags = operand_def(RFLAGSRegisterType)

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
        return (then_label.label,)

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


@irdl_op_definition
class S_JaOp(ConditionalJumpOperation):
    """
    Jump if above (CF=0 and ZF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.ja"


@irdl_op_definition
class S_JaeOp(ConditionalJumpOperation):
    """
    Jump if above or equal (CF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jae"


@irdl_op_definition
class S_JbOp(ConditionalJumpOperation):
    """
    Jump if below (CF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jb"


@irdl_op_definition
class S_JbeOp(ConditionalJumpOperation):
    """
    Jump if below or equal (CF=1 or ZF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jbe"


@irdl_op_definition
class S_JcOp(ConditionalJumpOperation):
    """
    Jump if carry (CF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jc"


@irdl_op_definition
class S_JeOp(ConditionalJumpOperation):
    """
    Jump if equal (ZF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.je"


@irdl_op_definition
class S_JgOp(ConditionalJumpOperation):
    """
    Jump if greater (ZF=0 and SF=OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jg"


@irdl_op_definition
class S_JgeOp(ConditionalJumpOperation):
    """
    Jump if greater or equal (SF=OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jge"


@irdl_op_definition
class S_JlOp(ConditionalJumpOperation):
    """
    Jump if less (SF≠OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jl"


@irdl_op_definition
class S_JleOp(ConditionalJumpOperation):
    """
    Jump if less or equal (ZF=1 or SF≠OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jle"


@irdl_op_definition
class S_JnaOp(ConditionalJumpOperation):
    """
    Jump if not above (CF=1 or ZF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jna"


@irdl_op_definition
class S_JnaeOp(ConditionalJumpOperation):
    """
    Jump if not above or equal (CF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnae"


@irdl_op_definition
class S_JnbOp(ConditionalJumpOperation):
    """
    Jump if not below (CF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnb"


@irdl_op_definition
class S_JnbeOp(ConditionalJumpOperation):
    """
    Jump if not below or equal (CF=0 and ZF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnbe"


@irdl_op_definition
class S_JncOp(ConditionalJumpOperation):
    """
    Jump if not carry (CF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnc"


@irdl_op_definition
class S_JneOp(ConditionalJumpOperation):
    """
    Jump if not equal (ZF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jne"


@irdl_op_definition
class S_JngOp(ConditionalJumpOperation):
    """
    Jump if not greater (ZF=1 or SF≠OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jng"


@irdl_op_definition
class S_JngeOp(ConditionalJumpOperation):
    """
    Jump if not greater or equal (SF≠OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnge"


@irdl_op_definition
class S_JnlOp(ConditionalJumpOperation):
    """
    Jump if not less (SF=OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnl"


@irdl_op_definition
class S_JnleOp(ConditionalJumpOperation):
    """
    Jump if not less or equal (ZF=0 and SF=OF).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnle"


@irdl_op_definition
class S_JnoOp(ConditionalJumpOperation):
    """
    Jump if not overflow (OF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jno"


@irdl_op_definition
class S_JnpOp(ConditionalJumpOperation):
    """
    Jump if not parity (PF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnp"


@irdl_op_definition
class S_JnsOp(ConditionalJumpOperation):
    """
    Jump if not sign (SF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jns"


@irdl_op_definition
class S_JnzOp(ConditionalJumpOperation):
    """
    Jump if not zero (ZF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jnz"


@irdl_op_definition
class S_JoOp(ConditionalJumpOperation):
    """
    Jump if overflow (OF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jo"


@irdl_op_definition
class S_JpOp(ConditionalJumpOperation):
    """
    Jump if parity (PF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jp"


@irdl_op_definition
class S_JpeOp(ConditionalJumpOperation):
    """
    Jump if parity even (PF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jpe"


@irdl_op_definition
class S_JpoOp(ConditionalJumpOperation):
    """
    Jump if parity odd (PF=0).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jpo"


@irdl_op_definition
class S_JsOp(ConditionalJumpOperation):
    """
    Jump if sign (SF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.js"


@irdl_op_definition
class S_JzOp(ConditionalJumpOperation):
    """
    Jump if zero (ZF=1).
    https://www.felixcloutier.com/x86/jcc
    """

    name = "x86.s.jz"


class RRROperation(Generic[R1InvT, R2InvT, R3InvT], IRDLOperation, X86Instruction, ABC):
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


@irdl_op_definition
class RRR_Vfmadd231pdOp(
    RRROperation[AVXRegisterType, AVXRegisterType, AVXRegisterType]
):
    """
    Multiply packed double-precision floating-point elements in r2 and r3, add the intermediate result to r1, and store the final result in r1.
    https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd
    """

    name = "x86.rrr.vfmadd231pd"


@irdl_op_definition
class RR_VmovapdOp(R_RR_Operation[AVXRegisterType, AVXRegisterType]):
    """
    Move aligned packed double precision floating-point values from zmm1 to zmm2 using writemask k1
    https://www.felixcloutier.com/x86/movapd
    """

    name = "x86.rr.vmovapd"


@irdl_op_definition
class MR_VmovapdOp(M_MR_Operation[GeneralRegisterType, AVXRegisterType]):
    """
    Move aligned packed double precision floating-point values from zmm1 to m512 using writemask k1
    https://www.felixcloutier.com/x86/movapd
    """

    name = "x86.mr.vmovapd"


@irdl_op_definition
class RM_VbroadcastsdOp(R_RM_Operation[AVXRegisterType, GeneralRegisterType]):
    """
    Broadcast low double precision floating-point element in m64 to eight locations in zmm1 using writemask k1
    https://www.felixcloutier.com/x86/vbroadcast
    """

    name = "x86.rm.vbroadcastsd"


class GetAnyRegisterOperation(Generic[R1InvT], IRDLOperation, X86Op, ABC):
    """
    This instruction allows us to create an SSAValue for a given register name.
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


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        if isinstance(op, FuncOp):
            print(f"{op.sym_name.data}:", file=output)
            continue
        assert isinstance(op, X86Op), f"{op}"
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)


def x86_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()
