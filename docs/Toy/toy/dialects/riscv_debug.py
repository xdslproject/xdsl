from __future__ import annotations

from xdsl.ir import Operation, SSAValue, Dialect

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    Operand,
    operand_def,
)

from xdsl.dialects import riscv


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "riscv_debug.print"

    rs: Operand = operand_def(riscv.RegisterType)

    def __init__(self, reg: SSAValue | Operation):
        super().__init__(operands=[reg])


RISCV_DEBUG = Dialect(
    [
        PrintOp,
    ],
    [],
)
