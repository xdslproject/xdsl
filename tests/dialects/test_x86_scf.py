from collections.abc import Callable

import pytest

from xdsl import ir
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import test, x86, x86_scf
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.traits import (
    EffectInstance,
    MemoryEffect,
    MemoryEffectKind,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    get_effects,
)
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("iter_arg_types", [(), (x86.registers.R11,)])
@pytest.mark.parametrize("gen_block", [True, False])
def test_for_rof_init(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
    iter_arg_types: tuple[ir.Attribute, ...],
    gen_block: bool,
):
    lb = create_ssa_value(x86.registers.R12)
    ub = create_ssa_value(x86.registers.R13)
    step = create_ssa_value(x86.registers.R10)
    operands = tuple(create_ssa_value(t) for t in iter_arg_types)

    if gen_block:
        body = ir.Block(
            arg_types=[x86.registers.R12, *iter_arg_types],
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
    assert op.body.block.arg_types == (x86.registers.R12, *iter_arg_types)
    with ImplicitBuilder(op.body) as (_i, *args):
        x86_scf.YieldOp(*args)
    op.verify()

    assert op.step is step


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("static_ub", [False, True])
@pytest.mark.parametrize("static_step", [False, True])
def test_for_rof_bounds_and_step(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
    static_ub: bool,
    static_step: bool,
):
    reg = x86.registers.UNALLOCATED_REG64
    lb = create_ssa_value(reg)
    ub_val = create_ssa_value(reg)
    step_val = create_ssa_value(reg)
    ub_attr = IntegerAttr(42, i64)
    step_attr = IntegerAttr(3, i64)

    ub = ub_attr if static_ub else ub_val
    step = step_attr if static_step else step_val

    op = loop_cls(
        lb,
        ub,
        step,
        (),
        ir.Block((x86_scf.YieldOp(),), arg_types=[reg]),
    )
    op.verify()

    assert (op.ub_attr is None) is (not static_ub)
    assert (op.ub_val is None) is static_ub
    assert (op.step_attr is None) is (not static_step)
    assert (op.step_val is None) is static_step

    assert op.ub is (ub_attr if static_ub else ub_val)
    assert op.step is (step_attr if static_step else step_val)


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


def test_effect_traits():
    """
    Check effects of operations in the x86_scf dialect.
    """
    operations = tuple(x86_scf.X86_Scf.operations)
    effects_ops = {op for op in operations if op.has_trait(MemoryEffect)}
    unknown_effects_ops = {op for op in operations if op not in effects_ops}

    # Sentinels to remind us to update this test when updating the dialect
    assert len(effects_ops) == 3
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
        x86_scf.ForOp,
        x86_scf.RofOp,
    }
    assert no_effects_ops == {
        x86_scf.YieldOp,
    }
