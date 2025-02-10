#!/usr/bin/env python3
"""Benchmark rewriting in xDSL."""

from xdsl.transforms.canonicalize import CanonicalizationRewritePattern
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations, region_dce


MODULE_OP = None


def time_pattern_rewriter__apply_patterns() -> None:
    """."""
    return
    pattern = GreedyRewritePatternApplier(
        [RemoveUnusedOperations(), CanonicalizationRewritePattern()]
    )
    PatternRewriteWalker(pattern, post_walk_func=region_dce).rewrite_module(
        MODULE_OP
    )


def time_pattern_rewriter__lower_dialect() -> None:
    """."""


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile  # type: ignore

    BENCHMARKS = {
        "time_pattern_rewriter__apply_patterns": time_pattern_rewriter__apply_patterns,
        "time_pattern_rewriter__lower_dialect": time_pattern_rewriter__lower_dialect,
    }
    profile(BENCHMARKS)
