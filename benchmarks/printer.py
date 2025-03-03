#!/usr/bin/env python3
"""Benchmarks for the printer of the xDSL implementation."""

from io import StringIO

from benchmarks.helpers import get_context, parse_module
from benchmarks.workloads import WorkloadBuilder
from xdsl.printer import Printer as XdslPrinter

CTX = get_context()
MODULE_PRINTER = XdslPrinter(stream=StringIO())


class Printer:
    """Benchmark the xDSL printer on MLIR files."""

    WORKLOAD_CONSTANT_100 = parse_module(CTX, WorkloadBuilder.constant_folding(100))
    WORKLOAD_CONSTANT_1000 = parse_module(CTX, WorkloadBuilder.constant_folding(1000))
    WORKLOAD_LARGE_DENSE_ATTR_HEX = parse_module(
        CTX, WorkloadBuilder.large_dense_attr_hex()
    )

    def time_constant_100(self) -> None:
        """Time lexing constant folding for 100 items."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_CONSTANT_100)

    def time_constant_1000(self) -> None:
        """Time lexing constant folding for 1000 items."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_CONSTANT_1000)

    def time_dense_attr_hex(self) -> None:
        """Time lexing a 1024x1024xi8 dense attribute given as a hex string."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_LARGE_DENSE_ATTR_HEX)


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    PRINTER = Printer()
    BENCHMARKS: dict[str, Callable[[], None]] = {
        "Printer.constant_100_input": PRINTER.time_constant_100,
        "Printer.constant_1000_input": PRINTER.time_constant_1000,
        "Printer.dense_attr_hex_input": PRINTER.time_dense_attr_hex,
    }
    profile(BENCHMARKS)
