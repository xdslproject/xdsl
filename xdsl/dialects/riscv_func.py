from __future__ import annotations

from typing import Annotated, Sequence

from xdsl.ir import Operation, SSAValue, Dialect, Attribute, Region
from xdsl.traits import HasParent
from xdsl.utils.exceptions import VerifyException

from xdsl.irdl import (
    IRDLOperation,
    OptOpAttr,
    OptOpResult,
    VarOpResult,
    VarOperand,
    irdl_op_definition,
    SingleBlockRegion,
    OpAttr,
)
from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.dialects import riscv


@irdl_op_definition
class SyscallOp(IRDLOperation):
    name = "riscv_func.syscall"
    args: Annotated[VarOperand, riscv.RegisterType]
    syscall_num: OpAttr[IntegerAttr[IntegerType]]
    result: Annotated[OptOpResult, riscv.RegisterType]

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
        if len(self.args) >= 7:
            raise VerifyException(
                f"Syscall op has too many operands ({len(self.args)}), expected fewer than 7"
            )
        if len(self.results) >= 3:
            raise VerifyException(
                f"Syscall op has too many results ({len(self.results)}), expected fewer than 3"
            )


@irdl_op_definition
class CallOp(IRDLOperation):
    """RISC-V function call operation"""

    name = "riscv_func.call"
    args: Annotated[VarOperand, riscv.RegisterType]
    func_name: OpAttr[StringAttr]
    ress: Annotated[VarOpResult, riscv.RegisterType]

    def __init__(
        self,
        func_name: StringAttr,
        args: Sequence[Operation | SSAValue],
        result_types: Sequence[riscv.RegisterType],
        comment: StringAttr | None = None,
    ):
        super().__init__(
            operands=[args],
            result_types=result_types,
            attributes={
                "func_name": func_name,
                "comment": comment,
            },
        )

    def verify_(self):
        if len(self.args) >= 9:
            raise VerifyException(
                f"Function op has too many operands ({len(self.args)}), expected fewer than 9"
            )

        if len(self.results) >= 3:
            raise VerifyException(
                f"Function op has too many results ({len(self.results)}), expected fewer than 3"
            )


@irdl_op_definition
class FuncOp(IRDLOperation):
    """RISC-V function definition operation"""

    name = "riscv_func.func"
    func_name: OpAttr[StringAttr]
    func_body: SingleBlockRegion

    def __init__(self, name: str, region: Region):
        attributes: dict[str, Attribute] = {"func_name": StringAttr(name)}

        super().__init__(attributes=attributes, regions=[region])


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """RISC-V function return operation"""

    name = "riscv_func.return"
    values: Annotated[VarOperand, riscv.RegisterType]
    comment: OptOpAttr[StringAttr]

    traits = frozenset([HasParent(FuncOp)])

    def __init__(
        self,
        values: Sequence[Operation | SSAValue],
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[values],
            attributes={
                "comment": comment,
            },
        )

    def verify_(self):
        if len(self.results) >= 3:
            raise VerifyException(
                f"Function op has too many results ({len(self.results)}), expected fewer than 3"
            )


RISCV_Func = Dialect(
    [
        SyscallOp,
        CallOp,
        FuncOp,
        ReturnOp,
    ],
    [],
)
