#!/usr/bin/env python3
"""Benchmark rewriting in xDSL."""

def time_op_creation__create() -> None:
    """."""

def time_op_creation__clone() -> None:
    """."""

if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile  # type: ignore

    BENCHMARKS: dict[str, Callable[[], None]] = {}
    profile(BENCHMARKS)
