from io import StringIO
from typing import Sequence

from xdsl.ir import OpResult
from xdsl.riscv_asm_writer import print_riscv_module
from xdsl.dialects import riscv, builtin
from xdsl.irdl import irdl_op_definition, IRDLOperation, Annotated, OpAttr
from xdsl.ir import SSAValue
from xdsl.dialects.builtin import IntegerType, IndexType, IntegerAttr
from xdsl.builder import Builder


@irdl_op_definition
class ExternalRiscvOp(IRDLOperation, riscv.RISCVPrinterInterface):
    name = "custom.custom_op"

    rd: Annotated[OpResult, riscv.RegisterType]

    imm: OpAttr[builtin.IntegerAttr[builtin.IntegerType]]

    def __init__(self, rd: str, imm: int = 0):
        super().__init__(
            result_types=[riscv.RegisterType(riscv.Register(rd))],
            attributes={"imm": builtin.IntegerAttr(imm, 64)},
        )

    def riscv_printed_name(self) -> str:
        return "custom.op"

    def riscv_printed_components(
        self,
    ) -> Sequence[
        IntegerAttr[IntegerType | IndexType] | riscv.LabelAttr | SSAValue | str | None
    ]:
        return (self.rd, self.imm)


def test_external_op_printing():
    @builtin.ModuleOp
    @Builder.implicit_region
    def module():
        riscv.LiOp(100, rd=riscv.Register("zero"))
        ExternalRiscvOp("zero", 101)

    io = StringIO()
    print_riscv_module(module, io)

    assert (
        io.getvalue()
        == """    li zero, 100
    custom.op zero, 101
"""
    )
