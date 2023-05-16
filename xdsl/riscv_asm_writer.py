from io import StringIO
from typing import IO
from xdsl.ir import Operation, SSAValue
from xdsl.dialects.riscv import RISCVOp, RegisterType, AnyIntegerAttr
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.utils.hints import isa

from xdsl.dialects import riscv


def print_riscv_module(module: ModuleOp, output: IO[str]):
    for op in module.ops:
        print_assembly_instruction(op, output)


def append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


def print_assembly_instruction(op: Operation, output: IO[str]) -> None:
    # default assembly code generator
    assert isinstance(op, RISCVOp)
    assert op.name.startswith("riscv.")
    instruction_name = op.name[6:]

    components: list[
        AnyIntegerAttr | riscv.LabelAttr | SSAValue | RegisterType | str | None
    ]
    comment: StringAttr | None

    match op:
        case riscv.GetRegisterOp():
            # Don't print assembly for creating a SSA value representing register
            return
        case riscv.CommentOp():
            desc = f"    # {op.comment.data}"
            print(desc, file=output)
            return
        case riscv.NullaryOperation():
            components = []
            comment = op.comment
        case riscv.RdRsRsOperation():
            components = [op.rd, op.rs1, op.rs2]
            comment = op.comment
        case riscv.RdRsImmOperation():
            components = [op.rd, op.rs1, op.immediate]
            comment = op.comment
        case riscv.RdImmOperation():
            components = [op.rd, op.immediate]
            comment = op.comment
        case riscv.RsRsImmOperation():
            components = [op.rs1, op.rs2, op.immediate]
            comment = op.comment
        case riscv.RsRsOffOperation():
            components = [op.rs1, op.rs2, op.offset]
            comment = op.comment
        case riscv.RdRsOperation():
            components = [op.rd, op.rs]
            comment = op.comment
        case riscv.CsrReadWriteImmOperation():
            components = [op.rd, op.csr, op.immediate]
            comment = op.comment
        case riscv.CsrReadWriteOperation():
            components = [op.rd, op.csr, op.rs1]
            comment = op.comment
        case riscv.CsrBitwiseImmOperation():
            components = [op.rd, op.csr, op.immediate]
            comment = op.comment
        case riscv.CsrBitwiseOperation():
            components = [op.rd, op.csr, op.rs1]
            comment = op.comment
        case riscv.JOp():
            # J op is a special case of JalOp with zero return register
            components = [op.immediate]
            comment = op.comment
        case riscv.RdImmJumpOperation():
            components = [op.rd, op.immediate]
            comment = op.comment
        case riscv.RdRsImmJumpOperation():
            components = [op.rd, op.rs1, op.immediate]
            comment = op.comment
        case _:
            raise ValueError(f"Unknown RISCV operation type :{type(op)}")

    component_strs: list[str] = []

    for component in components:
        if component is None:
            continue
        elif isa(component, AnyIntegerAttr):
            component_strs.append(f"{component.value.data}")
        elif isinstance(component, riscv.LabelAttr):
            component_strs.append(component.data)
        elif isinstance(component, str):
            component_strs.append(component)
        elif isinstance(component, RegisterType):
            component_strs.append(component.register_name)
        else:
            assert isinstance(component.typ, RegisterType)
            reg = component.typ.register_name
            component_strs.append(reg)

    code = f"    {instruction_name}"
    if len(component_strs):
        code += f" {', '.join(component_strs)}"
    code = append_comment(code, comment)
    print(code, file=output)


def riscv_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_riscv_module(module, stream)
    return stream.getvalue()
