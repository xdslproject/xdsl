from dataclasses import dataclass
from typing import IO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.target import Target


def print_to_mps(prog: ModuleOp, output: IO[str]) -> None:
    """
    Lower an MLIR module to the proprietary Air Intermediate Representation (AIR).

    This function emits the AIR bitcode required to target Apple GPUs directly,
    bypassing the Metal Shading Language source generation step.

    Args:
        prog: The MLIR ModuleOp to be lowered.
        output: The IO stream (e.g., stdout or a file) to write the generated bitcode to.

    Returns:
        None. The output is written directly to the provided stream.
    """
    raise NotImplementedError("MPS backend not yet implemented")


@dataclass(frozen=True)
class MPSTarget(Target):
    name = "mps"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        print_to_mps(module, output)
