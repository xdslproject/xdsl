#!/usr/bin/env python3
"""Benchmark running xDSL opt end-to-end on MLIR files."""

from pathlib import Path

from xdsl.xdsl_opt_main import xDSLOptMain

BENCHMARKS_DIR = Path(__file__).parent
RAW_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "raw_test_mlir"
EXTRA_MLIR_DIR = BENCHMARKS_DIR / "resources" / "extra_mlir"
MLIR_FILES: dict[str, Path] = {
    "empty_program": RAW_TEST_MLIR_DIR / "xdsl_opt__empty_program.mlir",
    "constant_folding": EXTRA_MLIR_DIR / "constant_folding_100.mlir",
    "constant_folding_long": EXTRA_MLIR_DIR / "constant_folding_1000.mlir",
    "loop_unrolling": EXTRA_MLIR_DIR / "loop_unrolling.mlir",
}


def time_end_to_end_opt__empty_program() -> None:
    """Time running the empty program."""
    runner = xDSLOptMain(
        args=[str(MLIR_FILES["empty_program"]), "-p", "constant-fold-interp"]
    )
    runner.run()  # type: ignore[no-untyped-call]


def time_end_to_end_opt__constant_folding() -> None:
    """Time running a constant folding example."""
    runner = xDSLOptMain(
        args=[str(MLIR_FILES["constant_folding"]), "-p", "canonicalize"]
    )
    runner.run()  # type: ignore[no-untyped-call]


def time_end_to_end_opt__constant_folding_long() -> None:
    """Time running a constant folding example."""
    runner = xDSLOptMain(
        args=[str(MLIR_FILES["constant_folding_long"]), "-p", "canonicalize"]
    )
    runner.run()  # type: ignore[no-untyped-call]


def time_end_to_end_opt__constant_folding_unverified() -> None:
    """Time running a constant folding example."""
    runner = xDSLOptMain(
        args=[str(MLIR_FILES["constant_folding"]), "-p", "canonicalize", "--disable-verify"]
    )
    runner.run()  # type: ignore[no-untyped-call]


def time_end_to_end_opt__loop_unrolling() -> None:
    """Time running a constant folding example."""
    runner = xDSLOptMain(
        args=[str(MLIR_FILES["loop_unrolling"]), "-p", "canonicalize"]
    )
    runner.run()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    from utils import profile

    BENCHMARKS = {
        "time_end_to_end_opt__empty_program": time_end_to_end_opt__empty_program,
        "time_end_to_end_opt__constant_folding": time_end_to_end_opt__constant_folding,
        "time_end_to_end_opt__constant_folding_unverified": time_end_to_end_opt__constant_folding_unverified,
        "time_end_to_end_opt__constant_folding_long": time_end_to_end_opt__constant_folding_long,
        "time_end_to_end_opt__loop_unrolling": time_end_to_end_opt__loop_unrolling,
    }
    profile(BENCHMARKS)
