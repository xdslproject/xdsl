from collections.abc import Callable

import pytest

from xdsl import ir
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import riscv, riscv_scf, test
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.riscv.attrs import I12, i12
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
@pytest.mark.parametrize(
    "step", [IntegerAttr(1, i12), create_ssa_value(riscv.Registers.A2)]
)
@pytest.mark.parametrize("iter_arg_types", [(), (riscv.Registers.T0,)])
@pytest.mark.parametrize("gen_block", [True, False])
def test_for_rof_init(
    loop_cls: type[riscv_scf.ForOp | riscv_scf.RofOp],
    step: IntegerAttr[I12] | ir.SSAValue,
    iter_arg_types: tuple[ir.Attribute, ...],
    gen_block: bool,
):
    lb = create_ssa_value(riscv.Registers.A0)
    ub = create_ssa_value(riscv.Registers.A1)
    operands = tuple(create_ssa_value(t) for t in iter_arg_types)

    if gen_block:
        body = ir.Block(
            arg_types=[riscv.Registers.A0, *iter_arg_types],
        )
    else:
        body = None

    op = loop_cls(
        lb,
        ub,
        step,
        operands,
        body,
    )
    assert op.body.block.arg_types == (riscv.Registers.A0, *iter_arg_types)
    with ImplicitBuilder(op.body) as (_i, *args):
        riscv_scf.YieldOp(*args)
    op.verify()

    if isinstance(step, ir.SSAValue):
        assert op.step_attr is None
        assert op.step_val is step
    else:
        assert op.step_attr is step
        assert op.step_val is None

    assert op.step is step


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
