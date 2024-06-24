from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.qref import QRefBase
from xdsl.dialects.qssa import QssaBase
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class QRefToQssaRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QRefBase, rewriter: PatternRewriter):
        # Create replacement operation
        new_op: QssaBase = op.ssa_op()
        # For gates there are no results to replace
        new_results = () if new_op.is_gate else new_op.results

        rewriter.replace_matched_op(new_op, new_results)

        if not new_op.is_gate:
            return

        # For gates we replace any other occurences of the original operands
        # with the results of the new operation

        old_operands = tuple(new_op.operands)

        # We do this by replacing all uses of the operand...
        for operand, result in zip(op.operands, new_op.results):
            operand.replace_by(result)

        # and then resetting the operands of the new operation
        new_op.operands = old_operands


class QRefToQssa(ModulePass):
    name = "qref-to-qssa"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            QRefToQssaRewritePattern(), apply_recursively=False
        ).rewrite_op(op)
