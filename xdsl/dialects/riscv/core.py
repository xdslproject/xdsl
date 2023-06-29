from __future__ import annotations

from abc import ABC, abstractmethod
from io import StringIO
from typing import IO, Iterable, Protocol, TypeAlias, runtime_checkable

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    StringAttr,
)
from xdsl.ir import Data, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    OptRegion,
    OptSingleBlockRegion,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    opt_region_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator
from xdsl.utils.hints import isa


@irdl_attr_definition
class SImm12Attr(IntegerAttr[IntegerType]):
    """
    A 12-bit immediate signed value.
    """

    name = "riscv.simm12"

    def __init__(self, value: int) -> None:
        super().__init__(value, IntegerType(12, Signedness.SIGNED))

    def verify(self) -> None:
        """
        All I- and S-type instructions with 12-bit signed immediates --- e.g., addi but not slli ---
        accept their immediate argument as an integer in the interval [-2048, 2047]. Integers in the subinterval [-2048, -1]
        can also be passed by their (unsigned) associates in the interval [0xfffff800, 0xffffffff] on RV32I,
        and in [0xfffffffffffff800, 0xffffffffffffffff] on both RV32I and RV64I.
        https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#signed-immediates-for-i--and-s-type-instructions
        """

        if 0xFFFFFFFFFFFFF800 <= self.value.data <= 0xFFFFFFFFFFFFFFFF:
            return

        if 0xFFFFF800 <= self.value.data <= 0xFFFFFFFF:
            return

        super().verify()


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "riscv.label"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(self.data)


class RISCVOp(Operation, ABC):
    """
    Base class for operations that can be a part of RISC-V assembly printing.
    """

    @abstractmethod
    def assembly_line(self) -> str | None:
        raise NotImplementedError()


# region Assembly printing


@runtime_checkable
class AssemblyInstructionArg(Protocol):
    def print_assembly_instruction_arg(self) -> str:
        ...


AssemblyInstructionArgType: TypeAlias = (
    AnyIntegerAttr | LabelAttr | SSAValue | AssemblyInstructionArg | str | None
)


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        assert isinstance(op, RISCVOp)
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()


def append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


def _assembly_line(
    name: str,
    args: Iterable[AssemblyInstructionArgType],
    comment: StringAttr | None = None,
    is_indented: bool = True,
) -> str:
    arg_strs: list[str] = []

    for arg in args:
        if arg is None:
            continue
        elif isa(arg, AnyIntegerAttr):
            arg_strs.append(f"{arg.value.data}")
        elif isinstance(arg, LabelAttr):
            arg_strs.append(arg.data)
        elif isinstance(arg, str):
            arg_strs.append(arg)
        elif isinstance(arg, AssemblyInstructionArg):
            arg_strs.append(arg.print_assembly_instruction_arg())
        elif isinstance(arg.typ, AssemblyInstructionArg):
            arg_strs.append(arg.typ.print_assembly_instruction_arg())

    code = "    " if is_indented else ""
    code += name
    if arg_strs:
        code += f" {', '.join(arg_strs)}"
    code = append_comment(code, comment)
    return code


class RISCVInstruction(RISCVOp):
    """
    Base class for operations that can be a part of RISC-V assembly printing. Must
    represent an instruction in the RISC-V instruction set, and have the following format:
    name arg0, arg1, arg2           # comment
    The name of the operation will be used as the RISC-V assembly instruction name.
    """

    comment: StringAttr | None = opt_attr_def(StringAttr)
    """
    An optional comment that will be printed along with the instruction.
    """

    @abstractmethod
    def assembly_line_args(self) -> tuple[AssemblyInstructionArgType, ...]:
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
        return _assembly_line(instruction_name, self.assembly_line_args(), self.comment)


@irdl_op_definition
class LabelOp(IRDLOperation, RISCVOp):
    """
    The label operation is used to emit text labels (e.g. loop:) that are used
    as branch, unconditional jump targets and symbol offsets.
    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#labels
    Optionally, a label can be associated with a single-block region, since
    that is a common target for jump instructions.
    For example, to generate this assembly:
    ```
    label1:
        add a0, a1, a2
    ```
    One needs to do the following:
    ``` python
    @Builder.implicit_region
    def my_add():
        a1_reg = TestSSAValue(riscv.RegisterType(riscv.Registers.A1))
        a2_reg = TestSSAValue(riscv.RegisterType(riscv.Registers.A2))
        riscv.AddOp(a1_reg, a2_reg, rd=riscv.Registers.A0)
    label_op = riscv.LabelOp("label1", my_add)
    ```
    """

    name = "riscv.label"
    label: LabelAttr = attr_def(LabelAttr)
    comment: StringAttr | None = opt_attr_def(StringAttr)
    data: OptRegion = opt_region_def("single_block")

    traits = frozenset([NoTerminator()])

    def __init__(
        self,
        label: str | LabelAttr,
        region: OptSingleBlockRegion = None,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(label, str):
            label = LabelAttr(label)
        if isinstance(comment, str):
            comment = StringAttr(comment)
        if region is None:
            region = Region()

        super().__init__(
            attributes={
                "label": label,
                "comment": comment,
            },
            regions=[region],
        )

    def assembly_line(self) -> str | None:
        return append_comment(f"{self.label.data}:", self.comment)


@irdl_op_definition
class DirectiveOp(IRDLOperation, RISCVOp):
    """
    The directive operation is used to emit assembler directives (e.g. .word; .text; .data; etc.)
    A more complete list of directives can be found here:
    https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#pseudo-ops
    """

    name = "riscv.directive"
    directive: StringAttr = attr_def(StringAttr)
    value: StringAttr | None = opt_attr_def(StringAttr)
    data: OptRegion = opt_region_def("single_block")

    traits = frozenset([NoTerminator()])

    def __init__(
        self,
        directive: str | StringAttr,
        value: str | StringAttr | None,
        region: OptSingleBlockRegion = None,
    ):
        if isinstance(directive, str):
            directive = StringAttr(directive)
        if isinstance(value, str):
            value = StringAttr(value)
        if region is None:
            region = Region()

        super().__init__(
            attributes={
                "directive": directive,
                "value": value,
            },
            regions=[region],
        )

    def assembly_line(self) -> str | None:
        if self.value is not None and self.value.data:
            value = self.value.data
        else:
            value = None

        return _assembly_line(self.directive.data, (value,), is_indented=False)


# endregion
