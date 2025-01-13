from typing import TypeAlias

from xdsl.dialects.arm.register import ARMRegisterType
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import SSAValue

AssemblyInstructionArg: TypeAlias = SSAValue


def append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


def assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isinstance(arg.type, ARMRegisterType):
        reg = arg.type.register_name
        return reg
    else:
        raise ValueError(f"Unexpected register type {arg.type}")


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
