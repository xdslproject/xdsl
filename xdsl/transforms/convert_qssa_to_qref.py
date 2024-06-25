from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.qssa import QssaBase
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ConvertQssaToQRefPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QssaBase, rewriter: PatternRewriter):
        # For gates the results of the original operation should be replaced by its operands
        new_results = op.operands if op.is_gate else None

        rewriter.replace_matched_op(op.ref_op(), new_results)


class ConvertQssaToQRef(ModulePass):
    name = "convert-qssa-to-qref"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertQssaToQRefPattern()).rewrite_op(op)
