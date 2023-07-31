from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypeVar

from xdsl.dialects import builtin
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

_canonicalize_passes = set()

_RewritePatternT = TypeVar("_RewritePatternT", bound=RewritePattern)


def is_canonicalization(pat: type[_RewritePatternT]) -> type[_RewritePatternT]:
    _canonicalize_passes.add(pat)
    return pat


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

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
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

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            CanonicalizationPattern([p() for p in _canonicalize_passes])
        ).rewrite_module(module)
