from collections.abc import Callable

import pytest

from xdsl import ir
from xdsl.dialects import riscv, riscv_scf, test
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.riscv.attrs import i12
from xdsl.traits import (
    EffectInstance,
    MemoryEffect,
    MemoryEffectKind,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    get_effects,
)
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("loop_cls", [riscv_scf.ForOp, riscv_scf.RofOp])
def test_for_rof_step(
    loop_cls: type[riscv_scf.ForOp | riscv_scf.RofOp],
):
    lb = create_ssa_value(riscv.Registers.A0)
    ub = create_ssa_value(riscv.Registers.A1)
    step_val = create_ssa_value(riscv.Registers.A2)
    step_attr = IntegerAttr(1, i12)

    op = loop_cls(
        lb,
        ub,
        step_val,
        (),
        ir.Block((riscv_scf.YieldOp(),), arg_types=[riscv.Registers.UNALLOCATED_INT]),
    )
    op.verify()

    assert op.step_attr is None
    assert op.step_val is step_val
    assert op.step is step_val

    op = loop_cls(
        lb,
        ub,
        step_attr,
        (),
        ir.Block((riscv_scf.YieldOp(),), arg_types=[riscv.Registers.UNALLOCATED_INT]),
    )
    op.verify()

    assert op.step_attr is step_attr
    assert op.step_val is None
    assert op.step is step_attr


@pytest.mark.parametrize("loop_cls", [riscv_scf.ForOp, riscv_scf.RofOp])
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
    loop_cls: type[riscv_scf.ForOp | riscv_scf.RofOp],
    op_factory: Callable[[], ir.Operation],
    effects: set[EffectInstance] | None,
):
    reg = riscv.Registers.UNALLOCATED_INT
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
                riscv_scf.YieldOp(),
            ),
            arg_types=[reg],
        ),
    )
    op.verify()

    assert get_effects(op) == effects


@pytest.mark.parametrize(
    "op_factory,effects",
    [
        (test.TestOp, None),
        (test.TestPureOp, set[MemoryEffect]()),
        (test.TestReadOp, {EffectInstance(MemoryEffectKind.READ)}),
        (test.TestWriteOp, {EffectInstance(MemoryEffectKind.WRITE)}),
    ],
)
def test_while_recursive_memory_effects(
    op_factory: Callable[[], ir.Operation],
    effects: set[EffectInstance] | None,
):
    reg = riscv.Registers.UNALLOCATED_INT
    arguments = (create_ssa_value(reg),)
    result_types = (reg,)

    op = riscv_scf.WhileOp(
        arguments,
        result_types,
        ir.Region(
            ir.Block(
                (riscv_scf.ConditionOp(create_ssa_value(reg)),),
                arg_types=[reg],
            )
        ),
        ir.Region(
            ir.Block(
                (
                    op_factory(),
                    riscv_scf.YieldOp(),
                ),
                arg_types=[reg],
            )
        ),
    )
    op.verify()
    assert get_effects(op) == effects


def test_effect_traits():
    """
    Check effects of operations in the riscv_scf dialect.
    """
    operations = tuple(riscv_scf.RISCV_Scf.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 5
    assert not unknown_effects_ops

    all_effects_trait_types = {
        type(trait)
        for op in effects_ops
        for trait in op.get_traits_of_type(MemoryEffect)
    }

    # Check below separately for each of these
    assert all_effects_trait_types == {
        RecursiveMemoryEffect,
        NoMemoryEffect,
    }

    recursive_effects_ops = {
        op for op in effects_ops if op.has_trait(RecursiveMemoryEffect)
    }
    no_effects_ops = {op for op in effects_ops if op.has_trait(NoMemoryEffect)}

    assert recursive_effects_ops == {
        riscv_scf.ForOp,
        riscv_scf.RofOp,
        riscv_scf.WhileOp,
    }
    assert no_effects_ops == {
        riscv_scf.YieldOp,
        riscv_scf.ConditionOp,
    }
