from __future__ import annotations

from typing import Callable

from xdsl.ir import Operation
from xdsl.ir.core import Operation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriting.query import Match, Query


class QueryRewritePattern(RewritePattern):
    query: Query
    rewrite: Callable[[Match, PatternRewriter], None]

    def __init__(
        self, query: Query, rewrite: Callable[[Match, PatternRewriter], None]
    ) -> None:
        self.query = query
        self.rewrite = rewrite

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if (match := self.query.match(op)) is not None:
            self.rewrite(match, rewriter)
