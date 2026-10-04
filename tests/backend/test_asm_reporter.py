import pytest

from xdsl.backend.block_throughput_cost_model import MCABlockThroughputCostModel
from xdsl.builder import Builder
from xdsl.dialects import x86_func

x86_reporter = MCABlockThroughputCostModel(
    target="x86_64-unknown-linux-gnu", arch="skylake"
)


@pytest.mark.skipif(
    not x86_reporter.is_available(),
    reason="llvm-mca is not installed or cannot analyse the x86 target",
)
def test_mca_reporter_x86():
    @Builder.implicit_region
    def trivial_x86_func():
        x86_func.RetOp()

    estimated_cost = x86_reporter.estimate_throughput(trivial_x86_func.block)
    assert estimated_cost is not None, (
        "MCA reporter should return a valid cost estimate"
    )
