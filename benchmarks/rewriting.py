#!/usr/bin/env python3
"""Benchmarks for the pattern rewriter of the xDSL implementation."""

from typing import Any

from benchmarks.helpers import get_context, parse_module
from benchmarks.workloads import WorkloadBuilder
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
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

    workload_constant_20: ModuleOp
    workload_constant_100: ModuleOp
    workload_constant_1000: ModuleOp

    def setup(self) -> None:
        """Setup the benchmarks."""
        self.setup_constant_folding_20()
        self.setup_constant_folding_100()
        self.setup_constant_folding_1000()

    def setup_constant_folding_20(self) -> None:
        """Setup the constant folding 20 items benchmark."""
        self.workload_constant_20 = PatternRewrite.WORKLOAD_CONSTANT_20.clone()

    def time_constant_folding_20(self) -> None:
        """Time canonicalizing constant folding for 20 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_20)

    def setup_constant_folding_100(self) -> None:
        """Setup the constant folding 100 items benchmark."""
        self.workload_constant_100 = PatternRewrite.WORKLOAD_CONSTANT_100.clone()

    def time_constant_folding_100(self) -> None:
        """Time canonicalizing constant folding for 100 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_100)

    def setup_constant_folding_1000(self) -> None:
        """Setup the constant folding 1000 items benchmark."""
        self.workload_constant_1000 = PatternRewrite.WORKLOAD_CONSTANT_1000.clone()

    def time_constant_folding_1000(self) -> None:
        """Time canonicalizing constant folding for 1000 items."""
        CANONICALIZE_PASS.apply(CTX, self.workload_constant_1000)


class RewritingMicrobenchmarks:
    def time_get_variadic_sizes(self) -> None:
        """Time getting the variadic size of an operation.

        (1/60th)
        """
        pass

    def time_populate_worklist(self) -> None:
        """."""
        # Walk region
        # Push to worklist

    def time_remove_unused_operation(self) -> None:
        """."""
        # is_trivially_dead

    def time_canonicalisation_pattern_rewrite(self) -> None:
        """."""
        # operation get_trait (other ubenchmark?)

    # pattern_rewriter_init

    # worklist pop

    # insertpoint before

    # ParameterisedAttributeInit

    # const_evaluate_operand

    # PatternRewriter.insert_op

    # SSAValue.replace_by

    # PatternRewriter.erase_op

    # result_only_effects

    # LiveSet.propagate_region_liveness

    # LiveSet.delete_dead

    # def time_(self) -> None:
    #     """."""


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    PATTERN_REWRITER = PatternRewrite()
    BENCHMARKS: dict[
        str, Callable[[], None] | tuple[Callable[[], None], Callable[[], Any]]
    ] = {
        "PatternRewriter.constant_folding_20": (
            PATTERN_REWRITER.time_constant_folding_20,
            PATTERN_REWRITER.setup_constant_folding_20,
        ),
        "PatternRewriter.constant_folding_100": (
            PATTERN_REWRITER.time_constant_folding_100,
            PATTERN_REWRITER.setup_constant_folding_100,
        ),
        "PatternRewriter.constant_folding_1000": (
            PATTERN_REWRITER.time_constant_folding_1000,
            PATTERN_REWRITER.setup_constant_folding_1000,
        ),
    }
    profile(BENCHMARKS)
