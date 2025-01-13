from typing import TypeAlias

from xdsl.dialects.arm.register import ARMRegisterType
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    StringAttr,
)
from xdsl.ir import (
    Data,
    SSAValue,
)
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "arm.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)


AssemblyInstructionArg: TypeAlias = (
    AnyIntegerAttr | LabelAttr | SSAValue | ARMRegisterType | str | int
)


def append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg, ARMRegisterType):
        return arg.name
    else:
        raise ValueError(f"Unexpected register type {type(arg)}")


def assembly_line(
    name: str,
    arg_str: str,
    comment: StringAttr | None = None,
    is_indented: bool = True,
) -> str:
    code = "    " if is_indented else ""
    code += name
    if arg_str:
        code += f" {arg_str}"
    code = append_comment(code, comment)
    return code
