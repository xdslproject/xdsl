import shutil

import pytest

from xdsl.backend.asm_perf_reporter import MCAReporter
from xdsl.builder import Builder
from xdsl.dialects import cost_model, x86_func

llvm_mca_available = shutil.which("llvm-mca") is not None


@pytest.mark.skipif(not llvm_mca_available, reason="llvm-mca is not installed")
def test_mca_reporter_x86():
    @Builder.implicit_region
    def trivial_x86_func():
        cost_model.BeginMCARegionOfInterestOp()
        x86_func.RetOp()
        cost_model.StopMCARegionOfInterestOp()

    arch = "skylake"
    reporter = MCAReporter(arch)
    estimated_cost = reporter.estimate_throughput(trivial_x86_func.block)
    assert estimated_cost is not None, (
        "MCA reporter should return a valid cost estimate"
    )
