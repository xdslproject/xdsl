#!/usr/bin/env python3
"""Benchmarks for the lexer of the xDSL implementation."""

from benchmarks.workloads import WorkloadBuilder
from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind


class Lexer:
    """Benchmark the xDSL lexer on MLIR files."""

    WORKLOAD_EMPTY = WorkloadBuilder.empty()
    WORKLOAD_CONSTANT_100 = WorkloadBuilder.constant_folding(100)
    WORKLOAD_CONSTANT_1000 = WorkloadBuilder.constant_folding(1000)
    WORKLOAD_LARGE_DENSE_ATTR = WorkloadBuilder.large_dense_attr()
    WORKLOAD_LARGE_DENSE_ATTR_HEX = WorkloadBuilder.large_dense_attr_hex()

    def time_empty_program(self) -> None:
        """Time lexing an empty program."""
        lexer_input = Input(Lexer.WORKLOAD_EMPTY, "empty")
        lexer = MLIRLexer(lexer_input)
        while lexer.lex().kind is not MLIRTokenKind.EOF:
            pass

    def time_constant_100(self) -> None:
        """Time lexing constant folding for 100 items."""
        lexer_input = Input(Lexer.WORKLOAD_CONSTANT_100, "constant_100")
        lexer = MLIRLexer(lexer_input)
        while lexer.lex().kind is not MLIRTokenKind.EOF:
            pass

    def time_constant_1000(self) -> None:
        """Time lexing constant folding for 1000 items."""
        lexer_input = Input(Lexer.WORKLOAD_CONSTANT_1000, "constant_1000")
        lexer = MLIRLexer(lexer_input)
        while lexer.lex().kind is not MLIRTokenKind.EOF:
            pass

    def ignore_time_dense_attr(self) -> None:
        """Time lexing a 1024x1024xi8 dense attribute."""
        lexer_input = Input(Lexer.WORKLOAD_LARGE_DENSE_ATTR, "dense_attr")
        lexer = MLIRLexer(lexer_input)
        while lexer.lex().kind is not MLIRTokenKind.EOF:
            pass

    def ignore_time_dense_attr_hex(self) -> None:
        """Time lexing a 1024x1024xi8 dense attribute given as a hex string."""
        lexer_input = Input(Lexer.WORKLOAD_LARGE_DENSE_ATTR_HEX, "dense_attr_hex")
        lexer = MLIRLexer(lexer_input)
        while lexer.lex().kind is not MLIRTokenKind.EOF:
            pass


if __name__ == "__main__":
    from bench_utils import Benchmark, profile

    LEXER = Lexer()
    profile(
        {
            "Lexer.empty_program": Benchmark(LEXER.time_empty_program),
            "Lexer.constant_100": Benchmark(LEXER.time_constant_100),
            "Lexer.constant_1000": Benchmark(LEXER.time_constant_1000),
            "Lexer.dense_attr": Benchmark(LEXER.ignore_time_dense_attr),
            "Lexer.dense_attr_hex": Benchmark(LEXER.ignore_time_dense_attr_hex),
        }
    )
