from collections.abc import Callable

import pytest

from xdsl import ir
from xdsl.dialects import test, x86, x86_scf
from xdsl.traits import EffectInstance, MemoryEffect, MemoryEffectKind, get_effects
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize(
    "op_factory,effects",
    [
        (test.TestOp, None),
        (test.TestPureOp, set[MemoryEffect]()),
        (test.TestReadOp, {EffectInstance(MemoryEffectKind.READ)}),
        (test.TestWriteOp, {EffectInstance(MemoryEffectKind.WRITE)}),
    ],
)
def test_for_rof_recursive_memory_effects(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
    op_factory: Callable[[], ir.Operation],
    effects: set[EffectInstance] | None,
):
    reg = x86.registers.UNALLOCATED_REG64
    lb = create_ssa_value(reg)
    ub = create_ssa_value(reg)
    step_val = create_ssa_value(reg)

    op = loop_cls(
        lb,
        ub,
        step_val,
        (),
        ir.Block(
            (
                op_factory(),
                x86_scf.YieldOp(),
            ),
            arg_types=[reg],
        ),
    )
    op.verify()

    assert get_effects(op) == effects
