from xdsl.dialects.riscv_cf import BranchOp, ConditionalBranchOperation, JOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value


class ElideConstantBranches(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, ConditionalBranchOperation):
            return

        rs1, rs2 = map(get_constant_value, (op.rs1, op.rs2))
        if rs1 is None or rs2 is None:
            return

        # check if the op would take the branch or not
        # TODO: take bitwidth into account
        branch_taken = op.const_evaluate(rs1.value.data, rs2.value.data, 32)

        # if branch is always taken, replace by jump
        if branch_taken:
            rewriter.replace_op(
                op,
                JOp(
                    op.then_arguments,
                    op.then_block,
                    comment=f"Constant folded {op.name}",
                ),
            )
        # if branch is never taken, replace by "fall through"
        else:
            rewriter.replace_op(
                op,
                BranchOp(
                    op.else_arguments,
                    op.else_block,
                    comment=f"Constant folded {op.name}",
                ),
            )
