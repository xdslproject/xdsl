#!/usr/bin/env python3
"""Benchmarks for the verifier of the xDSL implementation."""

from benchmarks.workloads import WorkloadBuilder


class Verifier:
    """Benchmark verification in xDSL.

    For a single rewriting pass, we verify with the input before the pass and
    the output after the pass.

    Note that this is run on the parsed input, rather than the output of the
    re-writing pass. This is because constant folding and dead-code elimination
    reduce the rewritten results to negligibly small sizes, so only the first
    verification pass contributes to the overall performance.
    """

    WORKLOAD_CONSTANT_100 = WorkloadBuilder.constant_folding_module(100)
    WORKLOAD_CONSTANT_1000 = WorkloadBuilder.constant_folding_module(1000)
    WORKLOAD_LARGE_DENSE_ATTR = WorkloadBuilder.large_dense_attr_module()

    def time_constant_100(self) -> None:
        """Time verifying constant folding for 100 items."""
        Verifier.WORKLOAD_CONSTANT_100.verify()

    def time_constant_1000(self) -> None:
        """Time verifying constant folding for 1000 items."""
        Verifier.WORKLOAD_CONSTANT_1000.verify()

    def time_dense_attr_hex(self) -> None:
        """Time verifying a 1024x1024xi8 dense attribute given as a hex string."""
        Verifier.WORKLOAD_LARGE_DENSE_ATTR.verify()


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    VERIFIER = Verifier()
    profile(
        {
            "Verifier.constant_100": Benchmark(VERIFIER.time_constant_100),
            "Verifier.constant_1000": Benchmark(VERIFIER.time_constant_1000),
            "Verifier.dense_attr_hex": Benchmark(VERIFIER.time_dense_attr_hex),
        }
    )
