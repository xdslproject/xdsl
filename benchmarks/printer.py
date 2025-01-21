#!/usr/bin/env python3
"""Benchmark printing in xDSL."""

# TODO: Define the benchmarks

if __name__ == "__main__":
    from collections.abc import Callable

    from utils import profile

    BENCHMARKS: dict[str, Callable[[], None]] = {}
    profile(BENCHMARKS)
