from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.dialects.stencil import (
    ApplyOp,
)
from xdsl.ir import (
    MLContext,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class StencilUnrollPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        pass


@dataclass
class StencilUnrollPass(ModulePass):
    name = "stencil-unroll"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([StencilUnrollPattern()])
        )
        walker.rewrite_module(op)
