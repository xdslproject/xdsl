from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.dialects.printf import PrintCharOp
from xdsl.ir.core import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerPrintCharToPutchar(RewritePattern):
    """
    Rewrite Pattern that rewrites printf.print_char to an
    external function call
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintCharOp, rewriter: PatternRewriter, /):
        func_call = func.Call.get("putchar", [op.char], [i32])
        # Add empty new_results, since a result is necessary for linking
        # putchar, but the result does not exist anywhere.
        rewriter.replace_matched_op(func_call, new_results=[])


class LowerPrintCharToPutcharPass(ModulePass):
    name = "lower-printchar-to-putchar"

    # lower to func.call
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerPrintCharToPutchar()]),
            apply_recursively=True,
        )
        walker.rewrite_module(op)
        # Add external putchar reference
        op.body.block.add_ops([func.FuncOp.external("putchar", [i32], [i32])])
