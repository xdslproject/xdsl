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
from xdsl.utils.exceptions import VerifyException


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

    def verify_(self):
        if (len(self.iter_args) + 1) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args)+1}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variable and loop-carried variables as arguments."
            )
        if self.body.block.args and (iter_var := self.body.block.args[0]):
            if not isinstance(iter_var.type, IntRegisterType):
                raise VerifyException(
                    f"The first block argument of the body is of type {iter_var.type}"
                    " instead of riscv.IntRegisterType"
                )
        for idx, arg in enumerate(self.iter_args):
            if self.body.block.args[idx + 1].type != arg.type:
                raise VerifyException(
                    f"Block arguments with wrong type, expected {arg.type}, "
                    f"got {self.body.block.args[idx].type}. Arguments after the "
                    f"induction variable must match the carried variables."
                )
        if len(self.body.ops) > 0 and isinstance(self.body.block.last_op, YieldOp):
            yieldop = self.body.block.last_op
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The riscv_scf.for must yield its carried variables."
                )
            for idx, arg in enumerate(yieldop.arguments):
                if self.iter_args[idx].type != arg.type:
                    raise VerifyException(
                        f"Expected {self.iter_args[idx].type}, got {arg.type}. The "
                        f"riscv_scf.for's riscv_scf.yield must match carried"
                        f"variables types."
                    )


RISCV_Scf = Dialect(
    [
        YieldOp,
        ForOp,
    ],
    [],
)
