#!/usr/bin/env python3
"""Benchmark rewriting in xDSL."""

from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.transforms.canonicalize import CanonicalizationRewritePattern
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations, region_dce

from xdsl.context import MLContext
from xdsl.parser import Parser


CTX = MLContext(allow_unregistered=True)

def parse_module(file: Path) -> ModuleOp:
    """Parse a file as a module."""
    return Parser(CTX, file.read_text()).parse_module()

BENCHMARKS_DIR = Path(__file__).parent
GENERIC_TEST_MLIR_DIR = BENCHMARKS_DIR / "resources" / "generic_test_mlir"
MLIR_FILES: dict[str, Path] = {
    "apply_pdl_extra_file": GENERIC_TEST_MLIR_DIR
    / "filecheck__transforms__apply-pdl__apply_pdl_extra_file.mlir",
    "add": GENERIC_TEST_MLIR_DIR
    / "filecheck__transforms__arith-add-immediate-zero.mlir",
}
PARSED_FILES: dict[str, ModuleOp] = {
    name: parse_module(file) for name, file in MLIR_FILES.items()
}


def time_pattern_rewriter__apply_patterns() -> None:
    """Time greedily pattern rewriting an operation."""
    pattern = GreedyRewritePatternApplier(
        [RemoveUnusedOperations(), CanonicalizationRewritePattern()]
    )
    PatternRewriteWalker(pattern, post_walk_func=region_dce).rewrite_module(
        PARSED_FILES["apply_pdl_extra_file"]
    )


def time_pattern_rewriter__lower_scf_to_cf() -> None:
    """Time lowering a module dialect."""
    lowering_pass = ConvertScfToCf()
    lowering_pass.apply(CTX, PARSED_FILES["apply_pdl_extra_file"])



if __name__ == "__main__":
    from bench_utils import profile  # type: ignore

    BENCHMARKS = {
        "time_pattern_rewriter__apply_patterns": time_pattern_rewriter__apply_patterns,
        "time_pattern_rewriter__lower_scf_to_cf": time_pattern_rewriter__lower_scf_to_cf,
    }
    profile(BENCHMARKS)
