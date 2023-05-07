from typing import IO
from xdsl.ir import Operation, SSAValue
from xdsl.dialects.riscv import RISCVOp, RegisterType, AnyIntegerAttr
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.hints import isa

from xdsl.dialects import riscv


def print_riscv_module(module: ModuleOp, output: IO[str]):
    for op in module.ops:
        print_assembly_instruction(op, output)


def print_assembly_instruction(op: Operation, output: IO[str]) -> None:
    # default assembly code generator
    assert isinstance(op, RISCVOp)
    assert op.name.startswith("riscv.")
    instruction_name = op.name[6:]

    components: list[AnyIntegerAttr | riscv.LabelAttr | SSAValue | str | None] = []

    match op:
        case riscv.GetRegisterOp():
            # Don't print assembly for creating a SSA value representing register
            return
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
        case riscv.RdRsOperation():
            components = [op.rd, op.rs]
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
            assert isinstance(component.typ, RegisterType)
            reg = component.typ.data.name
            if reg is None:
                raise ValueError("Cannot emit riscv assembly for unallocated register")
            component_strs.append(reg)

    code = f"    {instruction_name} {', '.join(component_strs)}"
    print(code, file=output)
