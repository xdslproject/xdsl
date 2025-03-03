#!/usr/bin/env python3
"""Benchmarks for the pattern rewriter of the xDSL implementation."""

from benchmarks.helpers import get_context, parse_module
from benchmarks.workloads import WorkloadBuilder
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.transforms.canonicalize import CanonicalizePass

CTX = get_context()
CTX.load_dialect(Arith)
CTX.load_dialect(Builtin)

CANONICALIZE_PASS = CanonicalizePass()


class PatternRewrite:
    """Benchmark rewriting in xDSL."""

    WORKLOAD_CONSTANT_20 = parse_module(CTX, WorkloadBuilder.constant_folding(20))
    WORKLOAD_CONSTANT_100 = parse_module(CTX, WorkloadBuilder.constant_folding(100))
    WORKLOAD_CONSTANT_1000 = parse_module(CTX, WorkloadBuilder.constant_folding(1_000))

    def time_constant_folding_20(self) -> None:
        """Time canonicalizing constant folding."""
        CANONICALIZE_PASS.apply(CTX, PatternRewrite.WORKLOAD_CONSTANT_20.clone())

    def time_constant_folding_100(self) -> None:
        """Time canonicalizing constant folding."""
        CANONICALIZE_PASS.apply(CTX, PatternRewrite.WORKLOAD_CONSTANT_100.clone())

    def time_constant_folding_1000(self) -> None:
        """Time canonicalizing constant folding."""
        CANONICALIZE_PASS.apply(CTX, PatternRewrite.WORKLOAD_CONSTANT_1000.clone())


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    PATTERN_REWRITER = PatternRewrite()
    BENCHMARKS: dict[str, Callable[[], None]] = {
        "PatternRewriter.constant_folding_20": PATTERN_REWRITER.time_constant_folding_20,
        "PatternRewriter.constant_folding_100": PATTERN_REWRITER.time_constant_folding_100,
        "PatternRewriter.constant_folding_1000": PATTERN_REWRITER.time_constant_folding_1000,
    }
    profile(BENCHMARKS)
