import pytest

from xdsl.context import Context
from xdsl.dialects.arith import AddiOp, Arith, ConstantOp
from xdsl.dialects.builtin import (
    Builtin,
    IntegerAttr,
    ModuleOp,
    i32,
)
from xdsl.ir import Operation
from xdsl.transforms.constant_folding_simple import ConstantFoldingSimplePass


@pytest.fixture(params=[(1,), (10, 100), (100, 100, 100)])
def constant_folding_workload(request: pytest.FixtureRequest) -> tuple[ModuleOp, int]:
    constants: tuple[int] = request.param
    ops: list[Operation] = []
    for i, constant in enumerate(constants):
        ops.append(ConstantOp(IntegerAttr(constant, i32)))
        if i > 0:
            ops.append(AddiOp(ops[-1], ops[-2]))
    return ModuleOp(ops), sum(constants)


def test_constant_folding_simple(
    constant_folding_workload: tuple[ModuleOp, int],
) -> None:
    """."""
    module, target = constant_folding_workload

    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    simple_pass = ConstantFoldingSimplePass()

    simple_pass.apply(ctx, module)
    result: int = module.ops.last.result.op.value.value.data
    assert result == target
