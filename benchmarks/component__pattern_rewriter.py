#!/usr/bin/env python3
"""Benchmark rewriting in xDSL."""

# TODO: Define the benchmarks

def time_pattern_rewriter__canonicalise() -> None:
    """Time rewriting a canonicalise pattern.

    Expect this to look something like `xdsl/xdsl/transforms/canonicalize.py`.
    """

if __name__ == "__main__":
    from collections.abc import Callable

    from utils import profile

    BENCHMARKS: dict[str, Callable[[], None]] = {}
    profile(BENCHMARKS)
