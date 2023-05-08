from io import StringIO
from typing import Sequence

from xdsl.builder import Builder
from xdsl.dialects import riscv, builtin
from xdsl.ir import OpResult, SSAValue
from xdsl.irdl import irdl_op_definition, IRDLOperation, Annotated, OpAttr
from xdsl.riscv_asm_writer import print_riscv_module, RISCVPrintableInterface


@irdl_op_definition
class ExternalRiscvOp(IRDLOperation, RISCVPrintableInterface):
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
        builtin.IntegerAttr[builtin.IntegerType | builtin.IndexType]
        | riscv.LabelAttr
        | SSAValue
        | str
        | int
        | None
    ]:
        return self.rd, self.imm.value.data


@irdl_op_definition
class ExternalRiscvCustomLineFormatOp(IRDLOperation, RISCVPrintableInterface):
    name = "custom.custom_line"

    rd: Annotated[OpResult, riscv.RegisterType]

    imm: OpAttr[builtin.IntegerAttr[builtin.IntegerType]]

    def __init__(self, rd: str, imm: int = 0):
        super().__init__(
            result_types=[riscv.RegisterType(riscv.Register(rd))],
            attributes={"imm": builtin.IntegerAttr(imm, 64)},
        )

    def riscv_print_line(self) -> str:
        assert isinstance(self.rd.typ, riscv.RegisterType)
        return f".custom-line {self.rd.typ.abi_name} {self.imm.value.data}"


def test_external_op_printing():
    @builtin.ModuleOp
    @Builder.implicit_region
    def module():
        riscv.LiOp(100, rd=riscv.Register("zero"))
        ExternalRiscvOp("zero", 101)
        ExternalRiscvCustomLineFormatOp("a0", 42)

    io = StringIO()
    print_riscv_module(module, io)

    assert (
        io.getvalue()
        == """    li zero, 100
    custom.op zero, 101
.custom-line a0 42
"""
    )
