from __future__ import annotations

from typing import Sequence

from xdsl.ir import Operation, SSAValue, Dialect, Attribute, Region
from xdsl.traits import HasParent, IsTerminator
from xdsl.utils.exceptions import VerifyException

from xdsl.irdl import (
    IRDLOperation,
    OptOpResult,
    VarOpResult,
    VarOperand,
    attr_def,
    irdl_op_definition,
    opt_attr_def,
    opt_result_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.dialects import riscv


@irdl_op_definition
class SyscallOp(IRDLOperation):
    name = "riscv_func.syscall"
    args: VarOperand = var_operand_def(riscv.RegisterType)
    syscall_num: IntegerAttr[IntegerType] = attr_def(IntegerAttr[IntegerType])
    result: OptOpResult = opt_result_def(riscv.RegisterType)

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
            result_types=[
                riscv.RegisterType(riscv.RISCVRegister()) if has_result else None
            ],
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
    args: VarOperand = var_operand_def(riscv.RegisterType)
    func_name: StringAttr = attr_def(StringAttr)
    ress: VarOpResult = var_result_def(riscv.RegisterType)

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
    func_name: StringAttr = attr_def(StringAttr)
    func_body: Region = region_def("single_block")

    def __init__(self, name: str, region: Region):
        attributes: dict[str, Attribute] = {"func_name": StringAttr(name)}

        super().__init__(attributes=attributes, regions=[region])


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """RISC-V function return operation"""

    name = "riscv_func.return"
    values: VarOperand = var_operand_def(riscv.RegisterType)
    comment: StringAttr | None = opt_attr_def(StringAttr)

    traits = frozenset([IsTerminator(), HasParent(FuncOp)])

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
