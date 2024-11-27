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


class ConvertQRefToQssaPattern(RewritePattern):
    """
    Replaces a qref operation by its qssa counterpart.
    If the operation is a gate, then subsequent uses of its operands should instead be given
    the results of the new operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: QRefBase, rewriter: PatternRewriter):
        # Create replacement operation
        new_op: QssaBase = op.ssa_op()
        # For gates there are no results to replace
        new_results = () if new_op.is_gate else new_op.results

        rewriter.replace_matched_op(new_op, new_results)

        if not new_op.is_gate:
            return

        # For gates we replace any other occurences of the original operands with the
        # results of the new operation, except for when used by the new operation.

        for operand, result in zip(op.operands, new_op.results):
            operand.replace_by_if(result, lambda use: use.operation is not new_op)


class ConvertQRefToQssa(ModulePass):
    """
    Converts uses of the qref dialect to the qssa dialect in a module.
    Inverse to the "convert-qssa-to-qref" pass.
    """

    name = "convert-qref-to-qssa"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertQRefToQssaPattern(), apply_recursively=False
        ).rewrite_module(op)
