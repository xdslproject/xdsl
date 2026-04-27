#!/usr/bin/env python3
from pathlib import Path

from bench_utils import Benchmark, profile

from xdsl.backend.llvm.convert import convert_module
from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.llvm import LLVM
from xdsl.dialects.vector import Vector
from xdsl.parser import Parser

FILECHECK_TEST = (
    Path(__file__).parents[1]
    / "tests"
    / "filecheck"
    / "backend"
    / "llvm"
    / "convert_op.mlir"
)


def _parse_module() -> ModuleOp:
    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(LLVM)
    ctx.load_dialect(Vector)
    module = Parser(ctx, FILECHECK_TEST.read_text()).parse_module()
    assert isinstance(module, ModuleOp)
    return module


class LLVMBackend:
    MODULE = _parse_module()

    def time_convert_module(self) -> None:
        convert_module(LLVMBackend.MODULE)


if __name__ == "__main__":
    BACKEND = LLVMBackend()
    profile(
        {
            "LLVMBackend.convert_module": Benchmark(BACKEND.time_convert_module),
        }
    )
