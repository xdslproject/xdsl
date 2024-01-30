from dataclasses import dataclass
from typing import Any

from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.riscv_cf import BranchOp, ConditionalBranchOperation, JOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern


class ElideConstantBranches(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, ConditionalBranchOperation):
            return

        module: Operation | None = op.parent_op()
        while not isinstance(module, ModuleOp) and module is not None:
            module = module.parent_op()

        if module is None:
            return

        # evaluate inputs to op
        inputs = const_evaluate_expr_inputs(op.rs1, op.rs2)

        # inputs are not compile time constant
        if not inputs.success:
            return

        assert inputs.value is not None
        assert len(inputs.value) == 2
        rs1, rs2 = inputs.value

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


@dataclass
class ConstEvalResult:
    value: tuple[Any, ...] | None

    @property
    def success(self) -> bool:
        return self.value is not None


def const_evaluate_expr_inputs(*exp_inputs: SSAValue) -> ConstEvalResult:
    """
    Tries to evaluate the inputs to an operatin by Evaluating ConstantLike and Pure operations
    using the interpreter.

    Returns either ConstEvalResult(None), or ConstEvalResult(inputs).

    If the interpretation fails, an InterpretationError is raised.
    """
    results: list[Any] = []
    for operand in exp_inputs:
        # block arguments cannot be argued about
        if not isinstance(operand.owner, Operation):
            return ConstEvalResult(None)

        # grab the value from Li ops:
        if isinstance(operand.owner, riscv.LiOp):
            imm = operand.owner.immediate
            if not isinstance(imm, IntegerAttr):
                return ConstEvalResult(None)
            results.append(imm.value.data)
        # propagate through mv ops
        elif isinstance(operand.owner, riscv.MVOp):
            evaluated_inputs = const_evaluate_expr_inputs(*operand.owner.operands)
            # check that the evaluation was successful
            if not evaluated_inputs.success:
                return ConstEvalResult(None)
            assert evaluated_inputs.value is not None
            results.extend(evaluated_inputs.value)
        else:
            # if op is neither ConstantLike nor Pure, fail
            return ConstEvalResult(None)

    return ConstEvalResult(tuple(results))
