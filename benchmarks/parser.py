#!/usr/bin/env python3
"""Benchmarks for the parser of the xDSL implementation."""

from benchmarks.workloads import WorkloadBuilder
from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.parser import Parser as XdslParser

CTX = Context(allow_unregistered=True)
CTX.load_dialect(Arith)
CTX.load_dialect(Builtin)
CTX.load_dialect(Func)


class Parser:
    """Benchmark the xDSL parser on MLIR files."""

    WORKLOAD_CONSTANT_100 = WorkloadBuilder.constant_folding(100)
    WORKLOAD_CONSTANT_1000 = WorkloadBuilder.constant_folding(1000)
    WORKLOAD_LARGE_DENSE_ATTR = WorkloadBuilder.large_dense_attr()
    WORKLOAD_LARGE_DENSE_ATTR_HEX = WorkloadBuilder.large_dense_attr_hex()
    WORKLOAD_LARGE_CONSTANT_TENSOR = str(
        WorkloadBuilder.large_constant_tensor((500, 500))
    )

    def time_constant_100(self) -> None:
        """Time parsing constant folding for 100 items."""
        XdslParser(CTX, Parser.WORKLOAD_CONSTANT_100).parse_module()

    def time_constant_1000(self) -> None:
        """Time parsing constant folding for 1000 items."""
        XdslParser(CTX, Parser.WORKLOAD_CONSTANT_1000).parse_module()

    def ignore_time_dense_attr(self) -> None:
        """Time parsing a 1024x1024xi8 dense attribute."""
        XdslParser(CTX, Parser.WORKLOAD_LARGE_DENSE_ATTR).parse_module()

    def ignore_time_dense_attr_hex(self) -> None:
        """Time parsing a 1024x1024xi8 dense attribute given as a hex string."""
        XdslParser(CTX, Parser.WORKLOAD_LARGE_DENSE_ATTR_HEX).parse_module()

    def time_large_constant_tensor(self) -> None:
        """Time parsing a large constant tensor."""
        XdslParser(CTX, Parser.WORKLOAD_LARGE_CONSTANT_TENSOR).parse_module()


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    PARSER = Parser()
    profile(
        {
            "Parser.constant_100": Benchmark(PARSER.time_constant_100),
            "Parser.constant_1000": Benchmark(PARSER.time_constant_1000),
            "Parser.dense_attr": Benchmark(PARSER.ignore_time_dense_attr),
            "Parser.dense_attr_hex": Benchmark(PARSER.ignore_time_dense_attr_hex),
            "Parser.large_constant_tensor": Benchmark(
                PARSER.time_large_constant_tensor
            ),
        }
    )
