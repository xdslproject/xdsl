from xdsl.dialects import linalg, memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.loop_nest_lowering_utils import rewrite_generic_to_loops


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter) -> None:
        if op.res:
            raise NotImplementedError(
                "lowering for linalg.generic with results not yet supported"
            )

        rewrite_generic_to_loops(rewriter, op, memref.Load.get, memref.Store.get)


class ConvertLinalgToLoopsPass(ModulePass):
    """
    Converts a linalg generic to perfectly nested loops.
    """

    name = "convert-linalg-to-loops"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
