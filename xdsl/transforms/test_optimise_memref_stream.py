from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms.test_lower_linalg_to_snitch import (
    OPTIMISE_MEMREF_STREAM_PASSES,
)


@dataclass(frozen=True)
class TestOptimiseMemRefStream(ModulePass):
    """
    A compiler pass used for testing the optimization of memref streams.
    """

    name = "test-optimise-memref-stream"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for p in OPTIMISE_MEMREF_STREAM_PASSES:
            p.apply(ctx, op)
