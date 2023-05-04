from typing import IO
from xdsl.ir import Operation
from xdsl.dialects.riscv import RISCVOp, RegisterType, AnyIntegerAttr
from xdsl.dialects.builtin import StringAttr, IntegerAttr, ModuleOp
from xdsl.utils.hints import isa


def print_riscv_module(module: ModuleOp, output: IO[str]):
    for op in module.ops:
        print_assembly_instruction(op, output)


def print_assembly_instruction(op: Operation, output: IO[str]) -> None:
    # default assembly code generator
    assert isinstance(op, RISCVOp)
    assert op.name.startswith("riscv.")
    instruction_name = op.name[6:]
    components: list[str] = []

    for result in op.results:
        assert isinstance(result.typ, RegisterType)
        assert isinstance(result.typ.data.name, str)
        components.append(result.typ.data.name)

    for operand in op.operands:
        assert isinstance(operand.typ, RegisterType)
        assert isinstance(operand.typ.data.name, str)
        components.append(operand.typ.data.name)

    if "offset" in op.attributes:
        label_attr = getattr(op, "offset")
        assert isinstance(label_attr, StringAttr)
        components.append(label_attr.data)

    if "immediate" in op.attributes:
        immediate_attr = getattr(op, "immediate")
        assert isinstance(immediate_attr, IntegerAttr)
        if isa(immediate_attr, AnyIntegerAttr):
            components.append(str(immediate_attr.value.data))

    if "comment" in op.attributes:
        comment = getattr(op, "comment")
        assert isinstance(comment, StringAttr)
    else:
        comment = None

    def append_comment(line: str, comment: StringAttr | None) -> str:
        if comment is None:
            return line

        padding = " " * max(0, 48 - len(line))

        return f"{line}{padding} # {comment.data}"

    code = f"    {instruction_name} {', '.join(components)}"
    print(append_comment(code, comment), file=output)
