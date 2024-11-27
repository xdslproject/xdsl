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
    """
    Replaces a qssa operation by its qref counterpart.
    Must rewire the results of the original operation if it is a gate.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QssaBase, rewriter: PatternRewriter):
        # For gates the results of the original operation should be replaced by its operands
        new_results = op.operands if op.is_gate else None

        rewriter.replace_matched_op(op.ref_op(), new_results)


class ConvertQssaToQRef(ModulePass):
    """
    Converts uses of the qssa dialect to the qref dialect in a module.
    Inverse to the "convert-qref-to-qssa" pass.
    """

    name = "convert-qssa-to-qref"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertQssaToQRefPattern()).rewrite_module(op)
