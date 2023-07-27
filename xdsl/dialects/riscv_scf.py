"""
RISC-V SCF dialect
"""
from __future__ import annotations

from typing import Sequence

from xdsl.dialects.riscv import IntRegisterType, RISCVRegisterType
from xdsl.ir import Dialect
from xdsl.irdl import (
    Block,
    IRDLOperation,
    Operand,
    Operation,
    Region,
    SSAValue,
    VarOperand,
    VarOpResult,
    irdl_op_definition,
    operand_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsTerminator, SingleBlockImplicitTerminator


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "riscv_scf.yield"

    arguments: VarOperand = var_operand_def(RISCVRegisterType)

    # TODO circular dependency disallows this set of traits
    # tracked by gh issues https://github.com/xdslproject/xdsl/issues/1218
    # traits = frozenset([HasParent((For, If, ParallelOp, While)), IsTerminator()])
    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(operands=[[SSAValue.get(operand) for operand in operands]])


@irdl_op_definition
class ForOp(IRDLOperation):
    name = "riscv_scf.for"

    lb: Operand = operand_def(IntRegisterType)
    ub: Operand = operand_def(IntRegisterType)
    step: Operand = operand_def(IntRegisterType)

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


RISCV_Scf = Dialect(
    [
        YieldOp,
        ForOp,
    ],
    [],
)
