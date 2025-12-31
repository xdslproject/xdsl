from typing import IO

from xdsl.dialects.builtin import ModuleOp


def print_to_mps(prog: ModuleOp, output: IO[str]):
    raise NotImplementedError("MPS backend not yet implemented")
