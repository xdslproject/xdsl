#!/usr/bin/env python3
"""Benchmarks for the parser of the xDSL implementation."""

from benchmarks.helpers import get_context
from benchmarks.workloads import WorkloadBuilder
from xdsl.parser import Parser as XdslParser

CTX = get_context()


class Parser:
    """Benchmark the xDSL parser on MLIR files."""

    WORKLOAD_CONSTANT_100 = WorkloadBuilder.constant_folding(100)
    WORKLOAD_CONSTANT_1000 = WorkloadBuilder.constant_folding(1000)
    WORKLOAD_LARGE_DENSE_ATTR = WorkloadBuilder.large_dense_attr()
    WORKLOAD_LARGE_DENSE_ATTR_HEX = WorkloadBuilder.large_dense_attr_hex()

    def time_constant_100(self) -> None:
        """Time lexing constant folding for 100 items."""
        XdslParser(CTX, Parser.WORKLOAD_CONSTANT_100).parse_module()

    def time_constant_1000(self) -> None:
        """Time lexing constant folding for 1000 items."""
        XdslParser(CTX, Parser.WORKLOAD_CONSTANT_1000).parse_module()

    def ignore_time_dense_attr(self) -> None:
        """Time lexing a 1024x1024xi8 dense attribute."""
        XdslParser(CTX, Parser.WORKLOAD_LARGE_DENSE_ATTR).parse_module()

    def time_dense_attr_hex(self) -> None:
        """Time lexing a 1024x1024xi8 dense attribute given as a hex string."""
        XdslParser(CTX, Parser.WORKLOAD_LARGE_DENSE_ATTR_HEX).parse_module()


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    PARSER = Parser()
    BENCHMARKS: dict[str, Callable[[], None]] = {
        "Parser.constant_100": PARSER.time_constant_100,
        "Parser.constant_1000": PARSER.time_constant_1000,
        # "Parser.dense_attr": PARSER.ignore_time_dense_attr,
        "Parser.dense_attr_hex": PARSER.time_dense_attr_hex,
    }
    profile(BENCHMARKS)
