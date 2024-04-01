from __future__ import annotations

from io import StringIO
from typing import IO

from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, ModuleOp, StringAttr
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    FloatRegisterType,
    IntRegisterType,
    LabelAttr,
    RISCVOp,
)
from xdsl.printer import Printer
from xdsl.utils.hints import isa


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isa(arg, AnyIntegerAttr):
        return f"{arg.value.data}"
    elif isinstance(arg, int):
        return f"{arg}"
    elif isinstance(arg, LabelAttr):
        return arg.data
    elif isinstance(arg, str):
        return arg
    elif isinstance(arg, IntRegisterType):
        return arg.register_name
    elif isinstance(arg, FloatRegisterType):
        return arg.register_name
    else:
        if isinstance(arg.type, IntRegisterType):
            reg = arg.type.register_name
            return reg
        elif isinstance(arg.type, FloatRegisterType):
            reg = arg.type.register_name
            return reg
        else:
            assert False, f"{arg.type}"


def append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


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


def print_immediate_value(printer: Printer, immediate: AnyIntegerAttr | LabelAttr):
    match immediate:
        case IntegerAttr():
            printer.print(immediate.value.data)
        case LabelAttr():
            printer.print_string_literal(immediate.data)


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        assert isinstance(op, RISCVOp), f"{op}"
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()
