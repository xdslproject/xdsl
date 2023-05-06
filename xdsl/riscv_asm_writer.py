from typing import IO, Sequence

from xdsl.ir import Operation, SSAValue
from xdsl.dialects.builtin import (
    ModuleOp,
    AnyIntegerAttr,
    IntegerAttr,
    IntegerType,
    IndexType,
)
from xdsl.utils.hints import isa

from xdsl.dialects import riscv


def print_riscv_module(module: ModuleOp, output: IO[str]):
    for op in module.ops:
        print_assembly_instruction(op, output)


def print_assembly_instruction(op: Operation, output: IO[str]) -> None:
    # allow riscv printable
    if isinstance(op, riscv.RISCVPrinterInterface):
        _print_component_strings(
            op.riscv_printed_name(), op.riscv_printed_components(), output
        )
        return

    # default assembly code generator
    assert isinstance(op, riscv.RISCVOp)
    assert op.name.startswith("riscv.")
    instruction_name = op.name[6:]

    components: list[AnyIntegerAttr | riscv.LabelAttr | SSAValue | str | None] = []

    match op:
        case riscv.NullaryOperation():
            pass
        case riscv.RdRsRsOperation():
            components = [op.rd, op.rs1, op.rs2]
        case riscv.RdRsImmOperation():
            components = [op.rd, op.rs1, op.immediate]
        case riscv.RdImmOperation():
            components = [op.rd, op.immediate]
        case riscv.RsRsImmOperation():
            components = [op.rs1, op.rs2, op.immediate]
        case riscv.RsRsOffOperation():
            components = [op.rs1, op.rs2, op.offset]
        case riscv.CsrReadWriteImmOperation():
            components = [op.rd, op.csr, op.immediate]
        case riscv.CsrReadWriteOperation():
            components = [op.rd, op.csr, op.rs1]
        case riscv.CsrBitwiseImmOperation():
            components = [op.rd, op.csr, op.immediate]
        case riscv.CsrBitwiseOperation():
            components = [op.rd, op.csr, op.rs1]
        case _:
            raise ValueError(f"Unknown RISCV operation type :{type(op)}")

    _print_component_strings(instruction_name, components, output)


def _print_component_strings(
    name: str,
    components: Sequence[
        IntegerAttr[IntegerType | IndexType] | riscv.LabelAttr | SSAValue | str | None
    ],
    output: IO[str],
):
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
        else:
            assert isinstance(component.typ, riscv.RegisterType)
            reg = component.typ.data.name
            if reg is None:
                raise ValueError("Cannot emit riscv assembly for unallocated register")
            component_strs.append(reg)

    code = f"    {name} {', '.join(component_strs)}"
    print(code, file=output)
