from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from io import StringIO
from typing import IO, Annotated, Generic, TypeAlias, TypeVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.hints import isa

from .register import GeneralRegisterType, X86RegisterType

R1InvT = TypeVar("R1InvT", bound=X86RegisterType)
R2InvT = TypeVar("R2InvT", bound=X86RegisterType)


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
    AnyIntegerAttr | SSAValue | GeneralRegisterType | str | int
)


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


class M_R_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one source register.
    """

    source = operand_def(R1InvT)

    def __init__(
        self,
        source: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source],
            attributes={
                "comment": comment,
            },
            result_types=[],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.source,)


@irdl_op_definition
class R_PushOp(M_R_Operation[GeneralRegisterType]):
    """
    Decreases %rsp and places r1 at the new memory location pointed to by %rsp.
    https://www.felixcloutier.com/x86/push
    """

    name = "x86.r.push"


class R_M_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one destination register.
    """

    rsp_input = operand_def(GeneralRegisterType("rsp"))
    destination = result_def(R1InvT)
    rsp_output = result_def(GeneralRegisterType("rsp"))

    def __init__(
        self,
        *,
        comment: str | StringAttr | None = None,
        rsp_input: Operation | SSAValue,
        destination: R1InvT,
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


@irdl_op_definition
class R_PopOp(R_M_Operation[GeneralRegisterType]):
    """
    Copies the value at the top of the stack into r1 and increases %rsp.
    https://www.felixcloutier.com/x86/pop
    """

    name = "x86.r.pop"


class R_R_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one register acting as both source and destination.
    """

    T = Annotated[GeneralRegisterType, ConstraintVar("T")]
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
class R_NotOp(R_R_Operation[GeneralRegisterType]):
    """
    bitwise not of r1, stored in r1
    x[r1] = ~x[r1]
    https://www.felixcloutier.com/x86/not
    """

    name = "x86.r.not"


class RMOperation(Generic[R1InvT, R2InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one register and one memory access with an optional offset.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

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
        return self.r1, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = _parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        if self.offset is not None:
            offset = _assembly_arg_str(self.offset)
            if self.offset.value.data > 0:
                return _assembly_line(
                    instruction_name,
                    f"{destination}, [{source}+{offset}]",
                    self.comment,
                )
            else:
                return _assembly_line(
                    instruction_name, f"{destination}, [{source}{offset}]", self.comment
                )
        else:
            return _assembly_line(
                instruction_name, f"{destination}, [{source}]", self.comment
            )


@irdl_op_definition
class RM_AddOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the value from the memory location pointed to by r2 to r1 and stores the result in r1.
    x[r1] = x[r1] + [x[r2]]
    https://www.felixcloutier.com/x86/add
    """

    name = "x86.rm.add"


@irdl_op_definition
class RM_SubOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Subtracts the value from the memory location pointed to by r2 from r1 and stores the result in r1.
    x[r1] = x[r1] - [x[r2]]
    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.rm.sub"


@irdl_op_definition
class RM_ImulOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the value from the memory location pointed to by r2 with r1 and stores the result in r1.
    x[r1] = x[r1] * [x[r2]]
    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.rm.imul"


@irdl_op_definition
class RM_AndOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r1 and [r2], stored in r1
    x[r1] = x[r1] & [x[r2]]
    https://www.felixcloutier.com/x86/and
    """

    name = "x86.rm.and"


@irdl_op_definition
class RM_OrOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r1 and [r2], stored in r1
    x[r1] = x[r1] | [x[r2]]
    https://www.felixcloutier.com/x86/or
    """

    name = "x86.rm.or"


@irdl_op_definition
class RM_XorOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r1 and [r2], stored in r1
    x[r1] = x[r1] ^ [x[r2]]
    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.rm.xor"


@irdl_op_definition
class RM_MovOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from the memory location pointed to by r2 into r1.
    x[r1] = [x[r2]]
    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.rm.mov"


class R_RI_Operation(Generic[R1InvT], IRDLOperation, X86Instruction, ABC):
    """
    A base class for x86 operations that have one register and an immediate value.
    """

    r1 = operand_def(R1InvT)
    immediate: AnyIntegerAttr = attr_def(AnyIntegerAttr)

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
        temp = _parse_optional_immediate_value(
            parser, IntegerType(32, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["immediate"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        _print_immediate_value(printer, self.immediate)
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
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

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
        destination = _assembly_arg_str(self.r1)
        if self.offset is not None:
            offset = _assembly_arg_str(self.offset)
            if self.offset.value.data > 0:
                destination = f"[{destination}+{offset}]"
            else:
                destination = f"[{destination}{offset}]"
        else:
            destination = f"[{destination}]"
        return destination, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = _parse_optional_immediate_value(
            parser, IntegerType(64, Signedness.SIGNED)
        )
        if temp is not None:
            attributes["offset"] = temp
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
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
    immediate: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

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
        destination = _assembly_arg_str(self.r1)
        immediate = _assembly_arg_str(self.immediate)
        if self.offset is not None:
            offset = _assembly_arg_str(self.offset)
            if self.offset.value.data > 0:
                destination = f"[{destination}+{offset}]"
            else:
                destination = f"[{destination}{offset}]"
        else:
            destination = f"[{destination}]"
        return destination, immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        temp = _parse_immediate_value(parser, IntegerType(64, Signedness.SIGNED))
        attributes["immediate"] = temp
        if parser.parse_optional_punctuation(",") is not None:
            temp2 = _parse_optional_immediate_value(
                parser, IntegerType(32, Signedness.SIGNED)
            )
            if temp2 is not None:
                attributes["offset"] = temp2
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        _print_immediate_value(printer, self.immediate)
        if self.offset is not None:
            printer.print(", ")
            _print_immediate_value(printer, self.offset)
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
    elif isinstance(arg, str):
        return arg
    elif isinstance(arg, GeneralRegisterType):
        return arg.register_name
    else:
        if isinstance(arg.type, GeneralRegisterType):
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


def _parse_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr:
    return parser.expect(
        lambda: _parse_optional_immediate_value(parser, integer_type),
        "Expected immediate",
    )


def _parse_optional_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr | None:
    """
    Parse an optional immediate value. If an integer is parsed, an integer attr with the specified type is created.
    """
    if (immediate := parser.parse_optional_integer()) is not None:
        return IntegerAttr(immediate, integer_type)
    if (immediate := parser.parse_optional_str_literal()) is not None:
        return LabelAttr(immediate)


def _print_immediate_value(printer: Printer, immediate: AnyIntegerAttr | LabelAttr):
    match immediate:
        case IntegerAttr():
            printer.print(immediate.value.data)
        case LabelAttr():
            printer.print_string_literal(immediate.data)


# endregion


class GetAnyRegisterOperation(Generic[R1InvT], IRDLOperation, X86Op):
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
