from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms.test_lower_linalg_to_snitch import (
    LOWER_SNITCH_STREAM_TO_ASM_PASSES,
)


@dataclass(frozen=True)
class TestLowerSnitchStreamToAsm(ModulePass):
    """
    A compiler pass used for testing of the lowering from ML kernels defined as
    snitch_stream + riscv operations to riscv-assembly leveraging Snitch cores.
    """

    name = "test-lower-snitch-stream-to-asm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for p in LOWER_SNITCH_STREAM_TO_ASM_PASSES:
            p.apply(ctx, op)
