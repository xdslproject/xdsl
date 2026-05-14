from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms.test_lower_linalg_to_snitch import (
    LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
)


@dataclass(frozen=True)
class TestLowerMemRefStreamToSnitchStream(ModulePass):
    """
    A compiler pass used for testing of the lowering from ML kernels defined as
    memref_stream to snitch_stream + riscv.
    """

    name = "test-lower-memref-stream-to-snitch-stream"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for p in LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES:
            p.apply(ctx, op)
