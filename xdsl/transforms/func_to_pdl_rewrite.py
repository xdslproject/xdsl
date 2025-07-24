"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, func, pdl, test
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class FuncToPdlRewritePattern(RewritePattern):
    """Rewrite pattern that transforms a function into a PDL rewrite operation."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: func.FuncOp | func.ReturnOp, rewriter: PatternRewriter
    ):
        if isinstance(op, func.FuncOp):
            pdl_root = test.TestPureOp(result_types=(pdl.OperationType(),))
            rewriter.insert_op_before_matched_op(pdl_root)
            op.detach_region(func_body := op.regions[0])

            rewrite_root = pdl_root.results[0]
            for arg in func_body.block.args:
                arg.replace_by(rewrite_root)
                func_body.block.erase_arg(arg)

            rewrite_op = pdl.RewriteOp(root=rewrite_root, body=func_body)
            rewriter.replace_matched_op(rewrite_op)
        else:
            rewriter.erase_matched_op()


class FuncToPdlRewrite(ModulePass):
    """
    A pass that transforms a function into a PDL rewrite operation.
    """

    name = "func-to-pdl-rewrite"

    def apply(self, ctx: Context | None, op: builtin.ModuleOp) -> None:
        """Apply the pass."""
        pattern = FuncToPdlRewritePattern()
        PatternRewriteWalker(pattern).rewrite_module(op)
