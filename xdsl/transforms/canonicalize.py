from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.traits import HasCanonicalisationPatternsTrait
from xdsl.transforms.dead_code_elimination import dce


class CanonicalizePass(ModulePass):
    """
    Applies all canonicalization patterns.
    """

    name = "canonicalize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        patterns = [
            pattern
            for ctx_op in ctx.registered_ops()
            if (trait := ctx_op.get_trait(HasCanonicalisationPatternsTrait)) is not None
            for pattern in trait.get_canonicalization_patterns()
        ]
        PatternRewriteWalker(GreedyRewritePatternApplier(patterns)).rewrite_module(op)
        dce(op)
