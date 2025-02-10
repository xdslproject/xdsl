#!/usr/bin/env python3
"""Benchmark the time to traverse xDSL IR."""

from xdsl.dialects.test import TestOp
from xdsl.ir import Block

EXAMPLE_BLOCK_NUM_OPS = 1_000
EXAMPLE_BLOCK = Block(ops=(TestOp() for _ in range(EXAMPLE_BLOCK_NUM_OPS)))


def time_ir_traversal__iterate_block_ops() -> None:
    """Time directly iterating over a block's operations."""
    for op in EXAMPLE_BLOCK.ops:
        assert op


def time_ir_traversal__walk_block_ops() -> None:
    """Time walking a block's operations."""
    for op in EXAMPLE_BLOCK.walk():
        assert op


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile  # type: ignore

    BENCHMARKS: dict[str, Callable[[], None]] = {}
    profile(BENCHMARKS)
