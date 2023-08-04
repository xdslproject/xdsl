from collections import defaultdict
from dataclasses import dataclass, field

from xdsl.dialects import builtin
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
)
from xdsl.transforms.dead_code_elimination import dce


@dataclass
class CanonicalizationPattern(GreedyRewritePatternApplier):
    """
    A variant on the GreedyRewritePatternApplier that applies at most
    n times per op to prevent infinite loops.
    """

    op_match_count: dict[Operation, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    repeat_apply_limit: int = field(default=5)

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # apply at most n times per op
        if self.op_match_count[op] > self.repeat_apply_limit:
            return
        self.op_match_count[op] += 1
        # invoke the greedy rewrite pattern applier
        super().match_and_rewrite(op, rewriter)


class CanonicalizationPass(ModulePass):
    """
    Applies all canonicalization patters.
    """

    name = "canonicalize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        patterns = [
            pattern
            for ctx_op in ctx.registered_ops()
            for pattern in ctx_op.get_canonicalization_patterns()
        ]
        PatternRewriteWalker(CanonicalizationPattern(patterns)).rewrite_module(op)
        dce(op)
