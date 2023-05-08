from __future__ import annotations

from typing import Annotated

from xdsl.ir import Operation, SSAValue, Dialect, Attribute, Region

from xdsl.irdl import (
    IRDLOperation,
    OptOpAttr,
    OptOpResult,
    OptOperand,
    VarOperand,
    irdl_op_definition,
    SingleBlockRegion,
    OpAttr,
)
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects import riscv
from xdsl.utils.exceptions import VerifyException


def opt_str_attr(attr: str | StringAttr | None) -> StringAttr | None:
    if attr is None:
        return None
    if isinstance(attr, StringAttr):
        return attr
    return StringAttr(attr)


def str_attr(attr: str | StringAttr) -> StringAttr:
    if isinstance(attr, StringAttr):
        return attr
    return StringAttr(attr)


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "riscv_func.call"
    args: Annotated[VarOperand, riscv.RegisterType]
    func_name: OpAttr[StringAttr]
    result: Annotated[OptOpResult, riscv.RegisterType]

    def __init__(
        self,
        func_name: StringAttr,
        args: list[Operation | SSAValue],
        has_result: bool = True,
        comment: StringAttr | None = None,
    ):
        super().__init__(
            operands=[args],
            result_types=[riscv.RegisterType(riscv.Register()) if has_result else None],
            attributes={
                "func_name": func_name,
                "comment": comment,
            },
        )


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "riscv_func.func"

    func_name: OpAttr[StringAttr]
    func_body: SingleBlockRegion

    def __init__(self, name: str, region: Region):
        attributes: dict[str, Attribute] = {"func_name": StringAttr(name)}

        super().__init__(attributes=attributes, regions=[region])

    def verify_(self):
        # Check that the returned value matches the type of the function
        if len(self.func_body.blocks) != 1:
            raise VerifyException("Expected FuncOp to contain one block")

        block = self.func_body.blocks[0]

        if not len(block.ops):
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.last_op

        if not isinstance(last_op, ReturnOp):
            raise VerifyException("Expected last op of FuncOp to be a ReturnOp")

        # Sasha: the following is copy/pasted code from the Toy dialect that checks
        # whether the type of the function matches the return. But the riscv FuncOp
        # doesn't have a function type, so I'm not sure whether one needs to be added,
        # or the ReturnOp needs to lose its operand.

        # operand = last_op.value
        # operand_typ = None if operand is None else operand.typ

        # return_typs = self.function_type.outputs.data

        # if len(return_typs):
        #     if len(return_typs) == 1:
        #         return_typ = return_typs[0]
        #     else:
        #         raise VerifyException(
        #             "Expected return type of func to have 0 or 1 values")
        # else:
        #     return_typ = None

        # if operand_typ != return_typ:
        #     raise VerifyException(
        #         "Expected return value to match return type of function")


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "riscv_func.return"
    value: Annotated[OptOperand, riscv.RegisterType]
    comment: OptOpAttr[StringAttr]

    def __init__(
        self,
        value: Operation | SSAValue | None = None,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)
        super().__init__(
            operands=[value],
            attributes={
                "comment": comment,
            },
        )


RISCV_FUNC = Dialect(
    [
        CallOp,
        FuncOp,
        ReturnOp,
    ],
    [],
)
