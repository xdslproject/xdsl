#!/usr/bin/env python3
"""Benchmark running xDSL opt end-to-end on MLIR files."""

from pathlib import Path

from xdsl.xdsl_opt_main import xDSLOptMain

BENCHMARKS_DIR = Path(__file__).parent
EXTRA_MLIR_DIR = BENCHMARKS_DIR / "resources" / "extra_mlir"


class ConstantFolding:
    """Benchmark running `xdsl-opt` on constant folding workloads."""

    WORKLOAD_100 = str(EXTRA_MLIR_DIR / "constant_folding_100.mlir")
    WORKLOAD_1000 = str(EXTRA_MLIR_DIR / "constant_folding_1000.mlir")

    def time_100(self) -> None:
        """Time constant folding for 100 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_100, "-p", "canonicalize"]).run()  # type: ignore[no-untyped-call]

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

    def time_100_none(self) -> None:
        """Time applying no optimisations for 100 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_100]).run()  # type: ignore[no-untyped-call]

    def ignore_time_1000(self) -> None:
        """Time constant folding for 1000 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_1000, "-p", "canonicalize"]).run()  # type: ignore[no-untyped-call]

    def ignore_time_1000_unverified(self) -> None:
        """Time constant folding for 1000 items without the verifier."""
        xDSLOptMain(
            args=[
                ConstantFolding.WORKLOAD_1000,
                "-p",
                "canonicalize",
                "--disable-verify",
            ]
        ).run()  # type: ignore[no-untyped-call]

    def ignore_time_1000_none(self) -> None:
        """Time applying no optimisations for 1000 items."""
        xDSLOptMain(args=[ConstantFolding.WORKLOAD_1000]).run()  # type: ignore[no-untyped-call]


class Miscellaneous:
    """Benchmark running `xdsl-opt` on miscellaneous workloads."""

    MLIR_FILES: dict[str, Path] = {
        "apply_pdl_extra_file": EXTRA_MLIR_DIR
        / "filecheck__transforms__apply-pdl__apply_pdl_extra_file.mlir",
        "arith-add-immediate-zero": EXTRA_MLIR_DIR
        / "filecheck__transforms__arith-add-immediate-zero.mlir",
        "large_dense_attr": EXTRA_MLIR_DIR / "large_dense_attr.mlir",
        "large_dense_attr_hex": EXTRA_MLIR_DIR / "large_dense_attr.mlir",
        "empty_program": EXTRA_MLIR_DIR / "xdsl_opt__empty_program.mlir",
        "loop_unrolling": EXTRA_MLIR_DIR / "loop_unrolling.mlir",
    }

    def time_empty_program(self) -> None:
        """Time running the empty program."""
        xDSLOptMain(
            args=[
                str(Miscellaneous.MLIR_FILES["empty_program"]),
                "-p",
                "constant-fold-interp",
            ]
        ).run()  # type: ignore[no-untyped-call]

    def time_loop_unrolling(self) -> None:
        """Time running a loop unrolling example."""
        xDSLOptMain(
            args=[str(Miscellaneous.MLIR_FILES["loop_unrolling"]), "-p", "canonicalize"]
        ).run()  # type: ignore[no-untyped-call]

    def time_apply_pdl_extra_file(self) -> None:
        """Time running the `apply_pdl_extra_file.mlir` file."""
        xDSLOptMain(
            args=[
                str(Miscellaneous.MLIR_FILES["apply_pdl_extra_file"]),
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]

    def time_add(self) -> None:
        """Time running the `arith-add-immediate-zero.mlir` file."""
        xDSLOptMain(
            args=[
                str(Miscellaneous.MLIR_FILES["arith-add-immediate-zero"]),
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]

    def ignore_time_dense_attr(self) -> None:
        """Time running a 1024x1024xi8 dense attribute."""
        xDSLOptMain(
            args=[
                str(Miscellaneous.MLIR_FILES["large_dense_attr"]),
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]

    def ignore_time_dense_attr_hex(self) -> None:
        """Time running a 1024x1024xi8 dense attribute given as a hex string."""
        xDSLOptMain(
            args=[
                str(Miscellaneous.MLIR_FILES["large_dense_attr_hex"]),
                "-p",
                "canonicalize",
            ]
        ).run()  # type: ignore[no-untyped-call]


class CIRCT:
    """Benchmark running `xdsl-opt` on CIRCT workloads."""

    ...


class ASL:
    """Benchmark running `xdsl-opt` on ASL workloads."""

    ...


if __name__ == "__main__":
    from bench_utils import profile  # type: ignore

    CONSTANT_FOLDING = ConstantFolding()
    MISCELLANEOUS = Miscellaneous()

    BENCHMARKS = {
        "ConstantFolding.100": CONSTANT_FOLDING.time_100,
        "ConstantFolding.100_unverified": CONSTANT_FOLDING.time_100_unverified,
        "ConstantFolding.100_none": CONSTANT_FOLDING.time_100_none,
        "ConstantFolding.1000": CONSTANT_FOLDING.ignore_time_1000,
        "ConstantFolding.1000_unverified": CONSTANT_FOLDING.ignore_time_1000_unverified,
        "ConstantFolding.1000_none": CONSTANT_FOLDING.ignore_time_1000_none,
        "Miscellaneous.empty_program": MISCELLANEOUS.time_empty_program,
        "Miscellaneous.loop_unrolling": MISCELLANEOUS.time_loop_unrolling,
        "Miscellaneous.apply_pdl_extra_file": MISCELLANEOUS.time_apply_pdl_extra_file,
        "Miscellaneous.add": MISCELLANEOUS.time_add,
        "Miscellaneous.dense_attr": MISCELLANEOUS.ignore_time_dense_attr,
        "Miscellaneous.dense_attr_hex": MISCELLANEOUS.ignore_time_dense_attr_hex,
    }
    profile(BENCHMARKS)
