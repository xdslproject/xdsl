#!/usr/bin/env python3
"""Benchmarks for the pipeline stages of the xDSL implementation."""

from pathlib import Path

from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

from pathlib import Path

from xdsl.context import MLContext
from xdsl.ir import Operation
from xdsl.parser import Parser as XdslParser

from xdsl.dialects.builtin import ModuleOp
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.transforms.canonicalize import CanonicalizationRewritePattern
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations, region_dce

from xdsl.context import MLContext


CTX = MLContext(allow_unregistered=True)

BENCHMARKS_DIR = Path(__file__).parent
GENERIC_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "generic_test_mlir"
RAW_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "raw_test_mlir"
EXTRA_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "extra_mlir"
MLIR_FILES: dict[str, Path] = {
    "apply_pdl_extra_file": GENERIC_TEST_MLIR_DIR
    / "filecheck__transforms__apply-pdl__apply_pdl_extra_file.mlir",
    "add": GENERIC_TEST_MLIR_DIR
    / "filecheck__transforms__arith-add-immediate-zero.mlir",
}

class Lexer:
    """Benchmark the xDSL lexer on MLIR files."""

    @classmethod
    def lex_input(cls, lexer_input: Input) -> None:
        """Lex an xDSL input."""
        lexer = MLIRLexer(lexer_input)
        while lexer.lex().kind is not MLIRTokenKind.EOF:
            pass

    @classmethod
    def lex_file(cls, mlir_file: Path) -> None:
        """Lex a mlir file."""
        print(mlir_file)
        contents = mlir_file.read_text()
        lexer_input = Input(contents, str(mlir_file))
        Lexer.lex_input(lexer_input)


    def time_apply_pdl_extra_file(self) -> None:
        """Time lexing the `apply_pdl_extra_file.mlir` file."""
        Lexer.lex_file(MLIR_FILES["apply_pdl_extra_file"])


    def time_all(self) -> None:
        """Time lexing all `.mlir` files in xDSL's `tests/` directory ."""
        mlir_files = RAW_TEST_MLIR_DIR.iterdir()
        for mlir_file in mlir_files:
            Lexer.lex_file(Path(mlir_file))


class Parser:
    """Benchmark the xDSL parser on MLIR files."""

    @classmethod
    def parse_input(cls, parser_input: str) -> Operation:
        """Parse a string."""
        parser = XdslParser(CTX, parser_input)
        return parser.parse_op()

    @classmethod
    def parse_file(cls, mlir_file: Path) -> Operation:
        """Parse a MLIR file."""
        contents = mlir_file.read_text()
        return Parser.parse_input(contents)

    def time_apply_pdl_extra_file(self) -> None:
        """Time parsing the `apply_pdl_extra_file.mlir` file."""
        Parser.parse_file(MLIR_FILES["apply_pdl_extra_file"])


    def time_add(self) -> None:
        """Time parsing the `add.mlir` file."""
        Parser.parse_file(MLIR_FILES["add"])


    def time_dense_attr(self) -> None:
        """Time parsing a 1024x1024xi8 dense attribute"""
        Parser.parse_file(EXTRA_TEST_MLIR_DIR / "large_dense_attr.mlir")


    def time_dense_attr_hex(self) -> None:
        """Time parsing a 1024x1024xi8 dense attribute given as a hex string"""
        Parser.parse_file(EXTRA_TEST_MLIR_DIR / "large_dense_attr_hex.mlir")


    def time_all(self) -> None:
        """Time parsing all `.mlir` files in xDSL's `tests/` directory ."""
        mlir_files = GENERIC_TEST_MLIR_DIR.iterdir()
        for mlir_file in mlir_files:
            Parser.parse_file(Path(mlir_file))


class PatternRewriter:
    """Benchmark rewriting in xDSL."""

    PARSED_FILES: dict[str, ModuleOp] = {
        name: XdslParser(CTX, file.read_text()).parse_module()
        for name, file in MLIR_FILES.items()
    }

    def time_apply_patterns(self) -> None:
        """Time greedily pattern rewriting an operation."""
        pattern = GreedyRewritePatternApplier(
            [RemoveUnusedOperations(), CanonicalizationRewritePattern()]
        )
        PatternRewriteWalker(pattern, post_walk_func=region_dce).rewrite_module(
            PatternRewriter.PARSED_FILES["apply_pdl_extra_file"]
        )

    def time_lower_scf_to_cf(self) -> None:
        """Time lowering a module dialect."""
        lowering_pass = ConvertScfToCf()
        lowering_pass.apply(CTX, PatternRewriter.PARSED_FILES["apply_pdl_extra_file"])


class Printer:
    """Benchmark printing in xDSL."""


if __name__ == "__main__":
    from collections.abc import Callable

    from bench_utils import profile

    LEXER = Lexer()
    PARSER = Parser()
    PATTERN_REWRITER = PatternRewriter()
    PRINTER = Printer()

    BENCHMARKS: dict[str, Callable[[], None]] = {
       "Lexer.apply_pdl_extra_file": LEXER.time_apply_pdl_extra_file,
       "Lexer.all": PARSER.time_all,
       "Parser.apply_pdl_extra_file": PARSER.time_apply_pdl_extra_file,
       "Parser.add": PARSER.time_add,
       "Parser.dense_attr": PARSER.time_dense_attr,
       "Parser.dense_attr_hex": PARSER.time_dense_attr_hex,
       "Parser.all": PARSER.time_all,
       "PatternRewriter.apply_patterns": PATTERN_REWRITER.time_apply_patterns,
       "PatternRewriter.lower_scf_to_cf": PATTERN_REWRITER.time_lower_scf_to_cf,
    }
    profile(BENCHMARKS)
