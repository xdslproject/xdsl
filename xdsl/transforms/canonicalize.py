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
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations, region_dce


class CanonicalizationRewritePattern(RewritePattern):
    """Rewrite pattern that applies a canonicalization pattern."""

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        traits = op.get_traits_of_type(HasCanonicalizationPatternsTrait)
        if not traits:
            return
        patterns = tuple(
            pattern for trait in traits for pattern in trait.get_patterns(type(op))
        )
        if len(patterns) == 1:
            patterns[0].match_and_rewrite(op, rewriter)
            return
        GreedyRewritePatternApplier(list(patterns)).match_and_rewrite(op, rewriter)


class CanonicalizePass(ModulePass):
    """
    Applies all canonicalization patterns.
    """

    name = "canonicalize"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pattern = GreedyRewritePatternApplier(
            [RemoveUnusedOperations(), CanonicalizationRewritePattern()]
        )
        PatternRewriteWalker(pattern, post_walk_func=region_dce).rewrite_module(op)
