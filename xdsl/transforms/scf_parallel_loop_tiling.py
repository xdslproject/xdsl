from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


@dataclass
class ScfParallelLoopTilingPass(ModulePass):
    name = "scf-parallel-loop-tiling"

    parallel_loop_tile_sizes: list[int]

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        pass
