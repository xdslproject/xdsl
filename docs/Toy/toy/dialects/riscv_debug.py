from __future__ import annotations

from xdsl.dialects import riscv
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import IRDLOperation, Operand, irdl_op_definition, operand_def


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "riscv_debug.print"

    rs: Operand = operand_def(riscv.IntRegisterType)

    def __init__(self, reg: SSAValue | Operation):
        super().__init__(operands=[reg])


RISCV_DEBUG = Dialect(
    [
        PrintOp,
    ],
    [],
)
