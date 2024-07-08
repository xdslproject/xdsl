from dataclasses import dataclass

from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import (
    ConvertFuncToRiscvFuncPass,
)
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemrefToRiscvPass
from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import ConvertScfToRiscvPass
from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.convert_memref_stream_to_snitch_stream import (
    ConvertMemrefStreamToSnitchStreamPass,
)
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass

TEST_LOWER_MEMREF_STREAM_TO_SNITCH_STREAM: tuple[ModulePass, ...] = (
    CanonicalizePass(),
    ConvertMemrefToRiscvPass(),
    LowerAffinePass(),
    ConvertScfToRiscvPass(),
    ConvertArithToRiscvPass(),
    ConvertFuncToRiscvFuncPass(),
    ConvertMemrefStreamToSnitchStreamPass(),
    ReconcileUnrealizedCastsPass(),
)


@dataclass(frozen=True)
class TestLowerMemrefStreamToSnitchStream(PipelinePass):
    """
    A compiler pass used for testing of the lowering from ML kernels defined as
    memref_stream to snitch_stream + riscv.
    """

    name = "test-lower-memref-stream-to-snitch-stream"

    passes: tuple[ModulePass, ...] = TEST_LOWER_MEMREF_STREAM_TO_SNITCH_STREAM
