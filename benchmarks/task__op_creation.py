#!/usr/bin/env python3
"""Benchmark rewriting in xDSL."""

# TODO: Define the benchmarks

if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile  # type: ignore

    BENCHMARKS: dict[str, Callable[[], None]] = {}
    profile(BENCHMARKS)
