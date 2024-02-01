from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
    ConvertRiscvScfToRiscvCfPass,
)
from xdsl.backend.riscv.lowering.convert_snitch_stream_to_snitch import (
    ConvertSnitchStreamToSnitch,
)
from xdsl.backend.riscv.lowering.reduce_register_pressure import (
    RiscvReduceRegisterPressurePass,
)
from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.lower_snitch import LowerSnitchPass
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass
from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation


class TestLowerLinalgToSnitchPass(ModulePass):
    """
    A compiler pass used for testing of the lowering from ML kernels defined as
    linalg.generic operations to riscv-assemby leveraging Snitch cores.
    """

    name = "test-lower-linalg-to-snitch"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PipelinePass(
            [
                SnitchRegisterAllocation(),
                ConvertSnitchStreamToSnitch(),
                LowerSnitchPass(),
                CanonicalizePass(),
                RiscvScfLoopRangeFoldingPass(),
                CanonicalizePass(),
                RiscvReduceRegisterPressurePass(),
                RISCVRegisterAllocation(),
                CanonicalizePass(),
                LowerRISCVFunc(),
                ConvertRiscvScfToRiscvCfPass(),
                CanonicalizePass(),
            ]
        ).apply(ctx, op)
