from dataclasses import dataclass

from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.convert_memref_stream_to_loops import (
    ConvertMemrefStreamToLoopsPass,
)
from xdsl.transforms.memref_stream_fold_fill import MemrefStreamFoldFillPass
from xdsl.transforms.memref_stream_generalize_fill import MemrefStreamGeneralizeFillPass
from xdsl.transforms.memref_stream_infer_fill import MemrefStreamInferFillPass
from xdsl.transforms.memref_stream_unnest_out_parameters import (
    MemrefStreamUnnestOutParametersPass,
)
from xdsl.transforms.memref_streamify import MemrefStreamifyPass
from xdsl.transforms.scf_for_loop_flatten import ScfForLoopFlattenPass

TEST_OPTIMISE_MEMREF_STREAM: tuple[ModulePass, ...] = (
    MemrefStreamInferFillPass(),
    MemrefStreamUnnestOutParametersPass(),
    MemrefStreamFoldFillPass(),
    MemrefStreamGeneralizeFillPass(),
    MemrefStreamifyPass(),
    ConvertMemrefStreamToLoopsPass(),
    CanonicalizePass(),
    ScfForLoopFlattenPass(),
)


@dataclass(frozen=True)
class TestOptimiseMemrefStream(PipelinePass):
    """
    A compiler pass used for testing the optimization of memref streams.
    """

    name = "test-optimise-memref-stream"

    passes: tuple[ModulePass, ...] = TEST_OPTIMISE_MEMREF_STREAM
