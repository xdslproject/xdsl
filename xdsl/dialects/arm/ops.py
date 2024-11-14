from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from typing import IO, Generic, TypeVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    ModuleOp,
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
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer

from .assembly import AssemblyInstructionArg, assembly_arg_str, assembly_line
from .register import GeneralRegisterType

R1InvT = TypeVar("R1InvT", bound=GeneralRegisterType)
R2InvT = TypeVar("R2InvT", bound=GeneralRegisterType)
R3InvT = TypeVar("R3InvT", bound=GeneralRegisterType)


class ARMOp(Operation, ABC):
    """
    Base class for operations that can be a part of ARM assembly printing.
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


class ARMInstruction(ARMOp):
    """
    Base class for operations that can be a part of ARM assembly printing. Must
    represent an instruction in the ARM instruction set.
    The name of the operation will be used as the ARM assembly instruction name.
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


class R_RR_Operation(Generic[R1InvT, R2InvT], IRDLOperation, ARMInstruction, ABC):
    """
    A base class for ARM operations that have two registers.
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
class RR_MovOp(R_RR_Operation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value of r1 into r2.
    x[r1] = x[r2]
    https://developer.arm.com/documentation/dui0473/m/arm-and-thumb-instructions/mov
    """

    name = "arm.rr.mov"


class GetAnyRegisterOperation(Generic[R1InvT], IRDLOperation, ARMOp):
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
    name = "arm.get_register"


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        if isinstance(op, FuncOp):
            print(f"{op.sym_name.data}:", file=output)
            continue
        assert isinstance(op, ARMOp), f"{op}"
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)
