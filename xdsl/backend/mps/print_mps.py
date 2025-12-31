from typing import IO

from xdsl.dialects.builtin import ModuleOp


def print_to_mps(prog: ModuleOp, output: IO[str]):
    # TODO: implement lowering to proprietary AIR format
    # see: https://github.com/sueszli/llvm-to-air
    # demo:
    # $ uv run xdsl-opt -t mps tests/filecheck/backend/mps/module.mlir
    # $ uv run lit tests/filecheck/backend/mps/module.mlir
    raise NotImplementedError("MPS backend not yet implemented")
