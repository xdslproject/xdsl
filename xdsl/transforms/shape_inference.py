from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import HasShapeInferencePatternsTrait


class ShapeInferenceRewritePattern(RewritePattern):
    """Rewrite pattern that applies a shape inference pattern."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        trait = op.get_trait(HasShapeInferencePatternsTrait)
        if trait is None:
            return
        patterns = trait.get_shape_inference_patterns()
        if len(patterns) == 1:
            patterns[0].match_and_rewrite(op, rewriter)
            return
        GreedyRewritePatternApplier(list(patterns)).match_and_rewrite(op, rewriter)


class ShapeInferencePass(ModulePass):
    """
    Applies all shape inference patterns.
    """

    name = "shape-inference"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ShapeInferenceRewritePattern()).rewrite_module(op)
