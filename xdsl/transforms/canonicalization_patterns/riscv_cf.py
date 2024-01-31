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

        # evaluate inputs to op
        inputs = const_evaluate_expr_inputs(op.rs1, op.rs2)

        # inputs are not compile time constant
        if inputs is None:
            return

        assert len(inputs) == 2
        rs1, rs2 = inputs

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


def const_evaluate_expr_inputs(*exp_inputs: SSAValue) -> tuple[Any, ...] | None:
    """
    Tries to evaluate the inputs to an operatin by Evaluating ConstantLike and
    Pure operations.

    Returns either None (cannot evaluate) or tuple[Any, ...] with the const
    evaluated values of the exp_inputs.
    """
    results: list[Any] = []
    for operand in exp_inputs:
        # block arguments cannot be argued about
        if not isinstance(operand.owner, Operation):
            return None

        # grab the value from Li ops:
        if isinstance(operand.owner, riscv.LiOp):
            imm = operand.owner.immediate
            if not isinstance(imm, IntegerAttr):
                return None
            results.append(imm.value.data)

        # propagate through mv ops
        elif isinstance(operand.owner, riscv.MVOp):
            evaluated_inputs = const_evaluate_expr_inputs(*operand.owner.operands)
            # check that the evaluation was successful
            if evaluated_inputs is None:
                return None
            results.extend(evaluated_inputs)
        else:
            # if op is neither ConstantLike nor Pure, fail
            return None

    return tuple(results)
