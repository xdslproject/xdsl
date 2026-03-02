from typing import IO

from xdsl.dialects.builtin import ModuleOp


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
