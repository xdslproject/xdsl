from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Concatenate,
)

from xdsl.ir import Operation
from xdsl.ir.core import Operation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriting.query import (
    Match,
    Query,
)
from xdsl.rewriting.query_builder import PatternQuery, QueryParams


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


@dataclass
class ConstraintConjunction:
    ...


def query_rewrite_pattern(
    query: PatternQuery[QueryParams],
) -> Callable[
    [Callable[Concatenate[PatternRewriter, QueryParams], None]], QueryRewritePattern
]:
    def impl(
        func: Callable[Concatenate[PatternRewriter, QueryParams], None]
    ) -> QueryRewritePattern:
        def rewrite(match: Match, rewriter: PatternRewriter) -> None:
            return func(rewriter, **match)  # pyright: ignore[reportGeneralTypeIssues]

        return QueryRewritePattern(query, rewrite)

    return impl
