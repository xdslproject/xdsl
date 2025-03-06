from xdsl.context import Context
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        infer_shapes(op)


def infer_shapes(op: builtin.ModuleOp):
    """
    A helper function for ShapeInferencePass which allows it to be called from
    within other passes while exposing the least restrictive API.
    """

    PatternRewriteWalker(ShapeInferenceRewritePattern()).rewrite_module(op)
