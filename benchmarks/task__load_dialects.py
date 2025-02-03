#!/usr/bin/env python3
"""Benchmark loading dialects in xDSL."""
import importlib

import xdsl.dialects.builtin
import xdsl.dialects.arith


def time_load_dialect__arith() -> None:
    """Time loading the `arith` dialect.

    Note that this must be done with `importlib.reload` rather than just
    directly importing with `from xdsl.dialects.arith import Arith` to avoid
    tests interacting with each other.
    """
    importlib.reload(xdsl.dialects.arith)


def time_load_dialect__builtin() -> None:
    """Time loading the `builtin` dialect."""
    importlib.reload(xdsl.dialects.builtin)


if __name__ == "__main__":
    from collections.abc import Callable

    from utils import profile

    BENCHMARKS: dict[str, Callable[[], None]] = {
        "time_load_dialect__arith": time_load_dialect__arith,
        "time_load_dialect__builtin": time_load_dialect__builtin,
    }
    profile(BENCHMARKS)
