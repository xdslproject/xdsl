#!/usr/bin/env python3
"""Benchmarks for the pipeline stages of the xDSL implementation.

This should live in its own file to avoid clobbering other benchmarks by
already having imported dependencies.
"""

import importlib

import xdsl
import xdsl.dialects.affine
import xdsl.dialects.arith
import xdsl.dialects.builtin
import xdsl.dialects.cf
import xdsl.dialects.test
import xdsl.interpreters.affine
import xdsl.interpreters.arith
import xdsl.interpreters.builtin
import xdsl.interpreters.cf
import xdsl.interpreters.func
import xdsl.interpreters.linalg
import xdsl.interpreters.memref
import xdsl.interpreters.memref_stream
import xdsl.interpreters.ml_program
import xdsl.interpreters.pdl
import xdsl.interpreters.printf
import xdsl.interpreters.riscv
import xdsl.interpreters.riscv_cf
import xdsl.interpreters.riscv_debug
import xdsl.interpreters.riscv_func
import xdsl.interpreters.riscv_libc
import xdsl.interpreters.riscv_scf
import xdsl.interpreters.riscv_snitch
import xdsl.interpreters.scf
import xdsl.interpreters.tensor


class ImportXDSL:
    """Benchmark importing xDSL."""

    def time_import_xdsl(self) -> None:
        """Time importing xDSL using the default asv mechanism."""
        importlib.reload(xdsl)


class ImportDialects:
    """Benchmark loading dialects in xDSL.

    Note that this must be done with `importlib.reload` rather than just
    directly importing with `from xdsl.dialects.arith import Arith` to avoid
    tests interacting with each other.
    """

    def ignore_time_affine_load(self) -> None:
        """Time loading the `arith` dialect."""
        importlib.reload(xdsl.dialects.affine)

    def ignore_time_arith_load(self) -> None:
        """Time loading the `arith` dialect."""
        importlib.reload(xdsl.dialects.arith)

    def ignore_time_builtin_load(self) -> None:
        """Time loading the `builtin` dialect."""
        importlib.reload(xdsl.dialects.builtin)

    def ignore_time_cf_load(self) -> None:
        """Time loading the `cf` dialect."""
        importlib.reload(xdsl.dialects.cf)

    def ignore_time_test_load(self) -> None:
        """Time loading the `test` dialect."""
        importlib.reload(xdsl.dialects.test)

    def time_all_constant_load(self) -> None:
        """Time all dialects used by the constant folding workload."""
        self.ignore_time_affine_load()
        self.ignore_time_arith_load()
        self.ignore_time_cf_load()


class ImportInterpreters:
    """Benchmark loading interpreters in xDSL."""

    def time_all_constant_load(self) -> None:
        """Time all interpreters used by the constant folding workload."""
        importlib.reload(xdsl.dialects.test)
        importlib.reload(xdsl.interpreters.affine)
        importlib.reload(xdsl.interpreters.tensor)
        importlib.reload(xdsl.interpreters.scf)
        importlib.reload(xdsl.interpreters.riscv_snitch)
        importlib.reload(xdsl.interpreters.riscv_scf)
        importlib.reload(xdsl.interpreters.riscv_libc)
        importlib.reload(xdsl.interpreters.riscv_func)
        importlib.reload(xdsl.interpreters.riscv_debug)
        importlib.reload(xdsl.interpreters.riscv_cf)
        importlib.reload(xdsl.interpreters.riscv)
        importlib.reload(xdsl.interpreters.printf)
        importlib.reload(xdsl.interpreters.pdl)
        importlib.reload(xdsl.interpreters.ml_program)
        importlib.reload(xdsl.interpreters.memref_stream)
        importlib.reload(xdsl.interpreters.memref)
        importlib.reload(xdsl.interpreters.linalg)
        importlib.reload(xdsl.interpreters.func)
        importlib.reload(xdsl.interpreters.cf)
        importlib.reload(xdsl.interpreters.builtin)
        importlib.reload(xdsl.interpreters.arith)


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    XDSL = ImportXDSL()
    DIALECTS = ImportDialects()
    INTERPRETERS = ImportInterpreters()
    profile(
        {
            "xDSL": Benchmark(XDSL.time_import_xdsl),
            "Dialects.affine_load": Benchmark(DIALECTS.ignore_time_affine_load),
            "Dialects.arith_load": Benchmark(DIALECTS.ignore_time_arith_load),
            "Dialects.builtin_load": Benchmark(DIALECTS.ignore_time_builtin_load),
            "Dialects.cf_load": Benchmark(DIALECTS.ignore_time_cf_load),
            "Dialects.test_load": Benchmark(DIALECTS.ignore_time_test_load),
            "Dialects.all_constant_load": Benchmark(DIALECTS.time_all_constant_load),
            "Interpreters.all_constant_load": Benchmark(
                INTERPRETERS.time_all_constant_load
            ),
        }
    )
