from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv_cf import BranchOp, ConditionalBranchOperation, JOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters import arith, riscv, riscv_debug
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.traits import ConstantLike, Pure
from xdsl.utils.exceptions import InterpretationError


class ElideConstantBranches(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, ConditionalBranchOperation):
            return

        module: Operation | None = op.parent_op()
        while not isinstance(module, ModuleOp) and module is not None:
            module = module.parent_op()

        if module is None:
            return

        # set up interpreter for constant evaluation
        try:
            interpreter = Interpreter(module)
            interpreter.register_implementations(riscv.RiscvFunctions())
            interpreter.register_implementations(riscv_debug.RiscvDebugFunctions())
            interpreter.register_implementations(arith.ArithFunctions())

            # evaluate inputs to op
            inputs = const_evaluate_expr_inputs(interpreter, op.rs1, op.rs2)
        except InterpretationError:
            return

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


def const_evaluate_expr_inputs(
    interp: Interpreter, *exp_inputs: SSAValue
) -> ConstEvalResult:
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

        # if constant like, it passes the check. Move on to next
        inputs: tuple[Any, ...]
        if operand.owner.has_trait(ConstantLike):
            inputs = ()
        # recurse on inputs of pure ops
        elif operand.owner.has_trait(Pure):
            evaluated_inputs = const_evaluate_expr_inputs(
                interp, *operand.owner.operands
            ).value
            if evaluated_inputs is None:
                return ConstEvalResult(None)
            inputs = evaluated_inputs
        else:
            # if op is neither ConstantLike nor Pure, fail
            return ConstEvalResult(None)

        res = interp.run_op(operand.owner, inputs)
        results.extend(res)

    return ConstEvalResult(tuple(results))
