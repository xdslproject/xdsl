import pytest

from xdsl.backend.block_throughput_cost_model import MCABlockThroughputCostModel
from xdsl.builder import Builder
from xdsl.dialects import x86_func

llvm_mca_available = MCABlockThroughputCostModel.is_installed()


@pytest.mark.skipif(not llvm_mca_available, reason="llvm-mca is not installed")
def test_mca_reporter_x86():
    @Builder.implicit_region
    def trivial_x86_func():
        x86_func.RetOp()

    reporter = MCABlockThroughputCostModel(
        target="x86_64-unknown-linux-gnu", arch="skylake"
    )
    estimated_cost = reporter.estimate_throughput(trivial_x86_func.block)
    assert estimated_cost is not None, (
        "MCA reporter should return a valid cost estimate"
    )
