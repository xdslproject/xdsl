#!/usr/bin/env python3
"""Benchmarks for the verifier of the xDSL implementation."""

from benchmarks.helpers import get_context, parse_module
from benchmarks.workloads import WorkloadBuilder

CTX = get_context()


class Verifier:
    """Benchmark verification in xDSL.

    For a single rewriting pass, we verify with the input before the pass and
    the output after the pass.

    Note that this is run on the parsed input, rather than the output of the
    re-writing pass. This is because constant folding and dead-code elimination
    reduce the rewritten results to negligibly small sizes, so only the first
    verification pass contributes to the overall performance.
    """

    WORKLOAD_CONSTANT_100 = parse_module(CTX, WorkloadBuilder.constant_folding(100))
    WORKLOAD_CONSTANT_1000 = parse_module(CTX, WorkloadBuilder.constant_folding(1000))
    WORKLOAD_LARGE_DENSE_ATTR_HEX = parse_module(
        CTX, WorkloadBuilder.large_dense_attr_hex()
    )

    def time_constant_100(self) -> None:
        """Time verifying constant folding for 100 items."""
        Verifier.WORKLOAD_CONSTANT_100.verify()

    def time_constant_1000(self) -> None:
        """Time verifying constant folding for 1000 items."""
        Verifier.WORKLOAD_CONSTANT_1000.verify()

    def time_dense_attr_hex(self) -> None:
        """Time verifying a 1024x1024xi8 dense attribute given as a hex string."""
        Verifier.WORKLOAD_LARGE_DENSE_ATTR_HEX.verify()


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    VERIFIER = Verifier()
    BENCHMARKS: dict[str, Callable[[], None]] = {
        "Verifier.constant_100": VERIFIER.time_constant_100,
        "Verifier.constant_1000": VERIFIER.time_constant_1000,
        "Verifier.dense_attr_hex": VERIFIER.time_dense_attr_hex,
    }
    profile(BENCHMARKS)
