from xdsl.dialects import builtin
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import HasCanonicalisationPatternsTrait
from xdsl.transforms.dead_code_elimination import dce


class CanonicalizationRewritePattern(RewritePattern):
    """Rewrite pattern that applies a canonicalization pattern."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        trait = op.get_trait(HasCanonicalisationPatternsTrait)
        if trait is None:
            return
        patterns = trait.get_canonicalization_patterns()
        if len(patterns) == 1:
            patterns[0].match_and_rewrite(op, rewriter)
            return
        GreedyRewritePatternApplier(list(patterns)).match_and_rewrite(op, rewriter)


class CanonicalizePass(ModulePass):
    """
    Applies all canonicalization patterns.
    """

    name = "canonicalize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(CanonicalizationRewritePattern()).rewrite_module(op)
        dce(op)
