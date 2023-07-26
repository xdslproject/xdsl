"""
RISC-V SCF dialect
"""
from __future__ import annotations

from typing import Sequence
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    Operand,
    operand_def,
    var_operand_def,
    AnyAttr,
    VarOperand,
    SSAValue,
    Operation,
    VarOpResult,
    var_result_def,
    Region,
    region_def,
    Block,
)
from xdsl.traits import SingleBlockImplicitTerminator, IsTerminator
from xdsl.ir import Dialect
from xdsl.dialects.riscv import RISCVRegisterType


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "rvscf.yield"

    arguments: VarOperand = var_operand_def(RISCVRegisterType)
    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class ForOp(IRDLOperation):
    name = "rvscf.for"

    lb: Operand = operand_def(RISCVRegisterType)
    ub: Operand = operand_def(RISCVRegisterType)
    step: Operand = operand_def(RISCVRegisterType)

    iter_args: VarOperand = var_operand_def(RISCVRegisterType)

    res: VarOpResult = var_result_def(RISCVRegisterType)

    body: Region = region_def("single_block")

    traits = frozenset([SingleBlockImplicitTerminator(YieldOp)])

    def __init__(
        self,
        lb: SSAValue | Operation,
        ub: SSAValue | Operation,
        step: SSAValue | Operation,
        iter_args: Sequence[SSAValue | Operation],
        body: Region | Sequence[Operation] | Sequence[Block] | Block,
    ):
        if isinstance(body, Block):
            body = [body]

        super().__init__(
            operands=[lb, ub, step, iter_args],
            result_types=[[SSAValue.get(a).type for a in iter_args]],
            regions=[body],
        )


RVSCF = Dialect(
    [
        YieldOp,
        ForOp,
    ],
    [],
)
