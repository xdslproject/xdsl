from abc import ABC
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


class RISCVPrintableInterface(ABC):
    """
    This interface is used so that other dialects can extend RISC-V printing
    without having to modify printing code or the risc-v dialect base.
    """

    def riscv_print_line(self) -> str:
        """
        Can be overwritten to write completely custom things to the output.
        """
        return RISCVPrintableInterface.format_riscv_instruction(
            self.riscv_printed_name(), self.riscv_printed_components()
        )

    def riscv_printed_name(self) -> str:
        """
        Give the name of the RISC-V instruction
        """
        raise NotImplemented

    def riscv_printed_components(
        self,
    ) -> Sequence[
        IntegerAttr[IntegerType | IndexType]
        | riscv.LabelAttr
        | SSAValue
        | str
        | int
        | None
    ]:
        """
        Return the list of "arguments" to the operation
        """
        raise NotImplemented

    @staticmethod
    def format_riscv_instruction(
        name: str,
        components: Sequence[
            IntegerAttr[IntegerType | IndexType]
            | riscv.LabelAttr
            | SSAValue
            | str
            | int
            | None
        ],
    ) -> str:
        """
        This method formats a RISC-V instruction.

        Given a name and a list of arguments, it correctly stringifies them and then
        prints the assembly line in a canonical format.
        """
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
            elif isinstance(component, int):
                component_strs.append(str(component))
            else:
                assert isinstance(component.typ, riscv.RegisterType)
                reg = component.typ.data.name
                if reg is None:
                    raise ValueError(
                        "Cannot emit riscv assembly for unallocated register"
                    )
                component_strs.append(reg)

        return f"    {name} {', '.join(component_strs)}"


def print_riscv_module(module: ModuleOp, output: IO[str]):
    for op in module.ops:
        print_assembly_instruction(op, output)


def print_assembly_instruction(op: Operation, output: IO[str]) -> None:
    # allow riscv printable
    if isinstance(op, RISCVPrintableInterface):
        print(op.riscv_print_line(), file=output)
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

    line = RISCVPrintableInterface.format_riscv_instruction(
        instruction_name, components
    )

    print(line, file=output)
