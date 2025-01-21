#!/usr/bin/env python3
"""Benchmark the xDSL parser on MLIR files."""

from pathlib import Path

from xdsl.context import MLContext
from xdsl.parser import Parser

BENCHMARKS_DIR = Path(__file__).parent
GENERIC_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "generic_test_mlir"
MLIR_FILES: dict[str, Path] = {
    "apply_pdl_extra_file": GENERIC_TEST_MLIR_DIR
    / "filecheck__transforms__apply-pdl__apply_pdl_extra_file.mlir",
    "add": GENERIC_TEST_MLIR_DIR
    / "filecheck__transforms__arith-add-immediate-zero.mlir",
}

CTX = MLContext(allow_unregistered=True)


def parse_input(parser_input: str) -> None:
    """Parse a string."""
    parser = Parser(CTX, parser_input)
    parser.parse_op()


def parse_file(mlir_file: Path) -> None:
    """Parse a MLIR file."""
    contents = mlir_file.read_text()
    parse_input(contents)


def time_parser__apply_pdl_extra_file() -> None:
    """Time parsing the `apply_pdl_extra_file.mlir` file."""
    parse_file(MLIR_FILES["apply_pdl_extra_file"])


def time_parser__add() -> None:
    """Time parsing the `add.mlir` file."""
    parse_file(MLIR_FILES["add"])


def time_parser__all() -> None:
    """Time parsing all `.mlir` files in xDSL's `tests/` directory ."""
    mlir_files = GENERIC_TEST_MLIR_DIR.iterdir()
    for mlir_file in mlir_files:
        parse_file(Path(mlir_file))


if __name__ == "__main__":
    from utils import profile

    BENCHMARKS = {
        "time_parser__apply_pdl_extra_file": time_parser__apply_pdl_extra_file,
        "time_parser__add": time_parser__add,
        "time_parser__all": time_parser__all,
    }
    profile(BENCHMARKS)
