#!/usr/bin/env python3
"""Benchmark the xDSL lexer on MLIR files."""

from pathlib import Path

from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

BENCHMARKS_DIR = Path(__file__).parent
RAW_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "raw_test_mlir"
MLIR_FILES: dict[str, Path] = {
    "apply_pdl_extra_file": RAW_TEST_MLIR_DIR
    / "filecheck__transforms__apply-pdl__apply_pdl_extra_file.mlir",
    "rvscf_lowering_emu": RAW_TEST_MLIR_DIR
    / "filecheck__with-riscemu__rvscf_lowering_emu.mlir",
}


def lex_input(lexer_input: Input) -> None:
    """Lex an xDSL input."""
    lexer = MLIRLexer(lexer_input)
    while lexer.lex().kind is not MLIRTokenKind.EOF:
        pass


def lex_file(mlir_file: Path) -> None:
    """Lex a mlir file."""
    contents = mlir_file.read_text()
    lexer_input = Input(contents, str(mlir_file))
    lex_input(lexer_input)


def time_lexer__apply_pdl_extra_file() -> None:
    """Time lexing the `apply_pdl_extra_file.mlir` file."""
    lex_file(MLIR_FILES["apply_pdl_extra_file"])


def time_lexer__rvscf_lowering_emu() -> None:
    """Time lexing the `rvscf_lowering_emu.mlir` file."""
    lex_file(MLIR_FILES["rvscf_lowering_emu"])


def time_lexer__all() -> None:
    """Time lexing all `.mlir` files in xDSL's `tests/` directory ."""
    mlir_files = RAW_TEST_MLIR_DIR.iterdir()
    for mlir_file in mlir_files:
        lex_file(Path(mlir_file))


if __name__ == "__main__":
    from utils import profile

    BENCHMARKS = {
        "time_lexer__apply_pdl_extra_file": time_lexer__apply_pdl_extra_file,
        "time_lexer__rvscf_lowering_emu": time_lexer__rvscf_lowering_emu,
        "time_lexer__all": time_lexer__all,
    }
    profile(BENCHMARKS)
