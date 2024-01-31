from typing import Any

from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.riscv_cf import BranchOp, ConditionalBranchOperation, JOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern


class ElideConstantBranches(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, ConditionalBranchOperation):
            return

        rs1, rs2 = map(const_evaluate_operand, (op.rs1, op.rs2))
        if rs1 is None or rs2 is None:
            return

        # check if the op would take the branch or not
        branch_taken = op.const_evaluate(rs1, rs2)

        # if branch is always taken, replace by jump
        if branch_taken:
            rewriter.replace_matched_op(
                JOp(
                    op.then_arguments,
                    op.then_block,
                    comment=f"Constant folded {op.name}",
                )
            )
        # if branch is never taken, replace by "fall through"
        else:
            rewriter.replace_matched_op(
                BranchOp(
                    op.else_arguments,
                    op.else_block,
                    comment=f"Constant folded {op.name}",
                )
            )


def const_evaluate_operand(operand: SSAValue) -> Any:
    """
    Try to constant evaluate an SSA value, returning None on failure.
    """
    if isinstance(operand.owner, riscv.LiOp):
        imm = operand.owner.immediate
        if not isinstance(imm, IntegerAttr):
            return None
        return imm.value.data
