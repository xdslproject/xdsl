#!/usr/bin/env python3
"""Benchmarks for the printer of the xDSL implementation."""

from benchmarks.workloads import WorkloadBuilder
from xdsl.printer import Printer as XdslPrinter

MODULE_PRINTER = XdslPrinter()


class Printer:
    """Benchmark the xDSL printer on MLIR files."""

    WORKLOAD_CONSTANT_0 = WorkloadBuilder.constant_folding_module(0)
    WORKLOAD_CONSTANT_100 = WorkloadBuilder.constant_folding_module(100)
    WORKLOAD_CONSTANT_1000 = WorkloadBuilder.constant_folding_module(1000)
    WORKLOAD_LARGE_DENSE_ATTR = WorkloadBuilder.large_dense_attr_module()

    def time_constant_0(self) -> None:
        """Time printing a constant folded workload."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_CONSTANT_0)

    def time_constant_100(self) -> None:
        """Time printing constant folding for 100 items."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_CONSTANT_100)

    def time_constant_1000(self) -> None:
        """Time printing constant folding for 1000 items."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_CONSTANT_1000)

    def time_dense_attr(self) -> None:
        """Time printing a 1024x1024xi8 dense attribute given as a hex string."""
        MODULE_PRINTER.print_op(Printer.WORKLOAD_LARGE_DENSE_ATTR)


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    PRINTER = Printer()
    profile(
        {
            "Printer.constant_0": Benchmark(PRINTER.time_constant_0),
            "Printer.constant_100": Benchmark(PRINTER.time_constant_100),
            "Printer.constant_1000": Benchmark(PRINTER.time_constant_1000),
            "Printer.dense_attr": Benchmark(PRINTER.time_dense_attr),
        }
    )
