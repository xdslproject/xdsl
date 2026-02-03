"""
A pass that applies the interpreter to operations with no side effects where all the
inputs are constant, replacing the computation with a constant value.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, func, pdl, test
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class FuncOpToPdlRewritePattern(RewritePattern):
    """Rewrite pattern that transforms a function into a PDL rewrite operation."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        pdl_root = test.TestPureOp(result_types=(pdl.OperationType(),))
        op.detach_region(func_body := op.regions[0])
        rewrite_root = pdl_root.results[0]
        for arg in func_body.block.args:
            arg.replace_by(rewrite_root)
            func_body.block.erase_arg(arg)

        pdl_pattern = pdl.PatternOp(
            benefit=1,
            sym_name=op.sym_name.data,
            body=Region(
                Block([pdl_root, pdl.RewriteOp(root=rewrite_root, body=func_body)])
            ),
        )
        rewriter.replace_op(op, pdl_pattern)


@dataclass
class ReturnOpToPdlRewritePattern(RewritePattern):
    """Rewrite pattern that erases return ops in PDL rewrite operations."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter):
        rewriter.erase_op(op)


class FuncToPdlRewrite(ModulePass):
    """
    A pass that transforms a function into a PDL rewrite operation.
    """

    name = "func-to-pdl-rewrite"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Apply the pass."""
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    FuncOpToPdlRewritePattern(),
                    ReturnOpToPdlRewritePattern(),
                ]
            )
        ).rewrite_module(op)
