#!/usr/bin/env python3
"""Benchmark running xDSL opt end-to-end on MLIR files."""

from contextlib import redirect_stdout
from functools import wraps
from os import devnull
from pathlib import Path

from xdsl.xdsl_opt_main import xDSLOptMain

BENCHMARKS_DIR = Path(__file__).parent
EXTRA_MLIR_DIR = BENCHMARKS_DIR / "resources" / "extra_mlir"
SUPPRESS_STDOUT = True


def suppress_stdout(func):
    """A decorator that suppresses stdout from a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not SUPPRESS_STDOUT:
            return func(*args, **kwargs)
        with open(devnull, "w") as fnull:
            with redirect_stdout(fnull):
                return func(*args, **kwargs)

    return wrapper


class ConstantFolding:
    """Benchmark running `xdsl-opt` on constant folding workloads."""

    WORKLOAD_4 = str(EXTRA_MLIR_DIR / "constant_folding_4.mlir")
    WORKLOAD_20 = str(EXTRA_MLIR_DIR / "constant_folding_20.mlir")
    WORKLOAD_100 = str(EXTRA_MLIR_DIR / "constant_folding_100.mlir")
    WORKLOAD_1000 = str(EXTRA_MLIR_DIR / "constant_folding_1000.mlir")

    @suppress_stdout
    def time_100(self) -> None:
        """Time constant folding for 100 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_100, "-p", "canonicalize"]).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def time_100_unverified(self) -> None:
        """Time constant folding for 100 items without the verifier."""
        xDSLOptMain(
            args=[
                ConstantFolding.WORKLOAD_100,
                "-p",
                "canonicalize",
                "--disable-verify",
            ]
        ).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def time_100_constant_fold_interp(self) -> None:
        """Time applying the `constant-fold-interp` pass for 100 items."""
        xDSLOptMain(
            args=[ConstantFolding.WORKLOAD_100, "-p", "constant-fold-interp"]
        ).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def time_100_none(self) -> None:
        """Time applying no optimisations for 100 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_100]).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def ignore_time_4(self) -> None:
        """Time constant folding for 4 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_4, "-p", "canonicalize"]).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def ignore_time_20(self) -> None:
        """Time constant folding for 20 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_20, "-p", "canonicalize"]).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def ignore_time_1000(self) -> None:
        """Time constant folding for 1000 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_1000, "-p", "canonicalize"]).run()  # type: ignore[no-untyped-call]


class Miscellaneous:
    """Benchmark running `xdsl-opt` on miscellaneous workloads."""

    WORKLOAD_EMPTY = str(EXTRA_MLIR_DIR / "xdsl_opt__empty_program.mlir")
    WORKLOAD_LARGE_DENSE_ATTR = str(EXTRA_MLIR_DIR / "large_dense_attr.mlir")
    WORKLOAD_LARGE_DENSE_ATTR_HEX = str(EXTRA_MLIR_DIR / "large_dense_attr_hex.mlir")

    @suppress_stdout
    def time_empty_program(self) -> None:
        """Time running the empty program."""
        xDSLOptMain(
            args=[
                Miscellaneous.WORKLOAD_EMPTY,
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def ignore_time_dense_attr(self) -> None:
        """Time running a 1024x1024xi8 dense attribute."""
        xDSLOptMain(
            args=[
                Miscellaneous.WORKLOAD_LARGE_DENSE_ATTR,
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]

    @suppress_stdout
    def time_dense_attr_hex(self) -> None:
        """Time running a 1024x1024xi8 dense attribute given as a hex string."""
        xDSLOptMain(
            args=[
                Miscellaneous.WORKLOAD_LARGE_DENSE_ATTR_HEX,
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]


class PDL:
    """Benchmark running `xdsl-opt` on PDL workloads."""

    ...


class CIRCT:
    """Benchmark running `xdsl-opt` on CIRCT workloads."""

    ...


class ASL:
    """Benchmark running `xdsl-opt` on ASL workloads."""

    ...


if __name__ == "__main__":
    from bench_utils import Benchmark, profile  # type: ignore

    CONSTANT_FOLDING = ConstantFolding()
    MISCELLANEOUS = Miscellaneous()

    profile(
        {
            "ConstantFolding.100": Benchmark(CONSTANT_FOLDING.time_100),
            "ConstantFolding.100_unverified": Benchmark(
                CONSTANT_FOLDING.time_100_unverified
            ),
            "ConstantFolding.100_constant_fold_interp": Benchmark(
                CONSTANT_FOLDING.time_100_constant_fold_interp
            ),
            "ConstantFolding.100_none": Benchmark(CONSTANT_FOLDING.time_100_none),
            "ConstantFolding.4": Benchmark(CONSTANT_FOLDING.ignore_time_4),
            "ConstantFolding.20": Benchmark(CONSTANT_FOLDING.ignore_time_20),
            "ConstantFolding.1000": Benchmark(CONSTANT_FOLDING.ignore_time_1000),
            "Miscellaneous.empty_program": Benchmark(MISCELLANEOUS.time_empty_program),
            "Miscellaneous.dense_attr": Benchmark(MISCELLANEOUS.ignore_time_dense_attr),
            "Miscellaneous.dense_attr_hex": Benchmark(
                MISCELLANEOUS.time_dense_attr_hex
            ),
        }
    )
