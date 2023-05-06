from __future__ import annotations

from typing import Annotated

from xdsl.ir import Operation, SSAValue, Dialect

from xdsl.irdl import (
    IRDLOperation,
    OptOpResult,
    VarOperand,
    irdl_op_definition,
    OpAttr,
)
from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, IntegerType
from xdsl.dialects import riscv


@irdl_op_definition
class SyscallOp(IRDLOperation):
    name = "riscv_ssa.syscall"
    args: Annotated[VarOperand, riscv.RegisterType]
    syscall_num: OpAttr[IntegerAttr[IntegerType]]
    result: Annotated[OptOpResult, riscv.RegisterType]
    """
    Some syscalls return values by putting them into a0. If result is not None, then the
    contents of a0 will be moved to its register.
    """

    def __init__(
        self,
        num: int | AnyIntegerAttr,
        has_result: bool = False,
        operands: list[SSAValue | Operation] = [],
    ):
        if isinstance(num, int):
            num = IntegerAttr.from_int_and_width(num, 32)
        super().__init__(
            operands=[operands],
            attributes={"syscall_num": num},
            result_types=[riscv.RegisterType(riscv.Register()) if has_result else None],
        )

    def verify_(self):
        assert len(self.args) < 7


RISCV_SSA = Dialect(
    [
        SyscallOp,
    ],
    [],
)
