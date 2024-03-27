from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
    ConvertRiscvScfToRiscvCfPass,
)
from xdsl.backend.riscv.lowering.convert_snitch_stream_to_snitch import (
    ConvertSnitchStreamToSnitch,
)
from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.convert_riscv_scf_for_to_frep import ConvertRiscvScfForToFrepPass
from xdsl.transforms.lower_snitch import LowerSnitchPass
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass
from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation


class TestLowerSnitchStreamToAsm(ModulePass):
    """
    A compiler pass used for testing of the lowering from ML kernels defined as
    snitch_stream + riscv operations to riscv-assemby leveraging Snitch cores.
    """

    name = "test-lower-snitch-stream-to-asm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PipelinePass(
            (
                ConvertRiscvScfForToFrepPass(),
                SnitchRegisterAllocation(),
                ConvertSnitchStreamToSnitch(),
                LowerSnitchPass(),
                CanonicalizePass(),
                RiscvScfLoopRangeFoldingPass(),
                CanonicalizePass(),
                RISCVRegisterAllocation(),
                CanonicalizePass(),
                ConvertRiscvScfToRiscvCfPass(),
                CanonicalizePass(),
            )
        ).apply(ctx, op)
