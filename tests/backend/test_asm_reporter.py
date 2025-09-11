from xdsl.backend.asm_perf_reporter import MCAReporter
from xdsl.builder import Builder
from xdsl.dialects import cost_model, x86_func
from xdsl.dialects.builtin import ModuleOp


def test_mca_reporter_x86():
    @ModuleOp
    @Builder.implicit_region
    def trivial_x86_func():
        cost_model.BeginMCARegionOfInterestOp()
        x86_func.RetOp()
        cost_model.StopMCARegionOfInterestOp()

    arch = "skylake"
    reporter = MCAReporter(arch, trivial_x86_func)
    estimated_cost = reporter.estimate_cost()
    assert estimated_cost is not None, (
        "MCA reporter should return a valid cost estimate"
    )
