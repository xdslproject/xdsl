from collections.abc import Callable

import pytest

from xdsl import ir
from xdsl.backend.liveness import VerifyLivenessContext
from xdsl.backend.register_allocator import live_ins_per_block
from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
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
    start = create_ssa_value(riscv.Registers.A0)
    stop = create_ssa_value(riscv.Registers.A1)
    operands = tuple(create_ssa_value(t) for t in iter_arg_types)

    if gen_block:
        body = ir.Block(
            arg_types=[start.type, *iter_arg_types],
        )
    else:
        body = None

    op = loop_cls(
        start,
        stop,
        step,
        operands,
        body,
    )
    assert op.start is start
    assert op.stop is stop
    assert tuple(op.operands[:2]) == (start, stop)
    assert op.body.block.arg_types == (start.type, *iter_arg_types)
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
def test_for_rof_allocate_bounds(
    loop_cls: type[riscv_scf.ForOp | riscv_scf.RofOp],
):
    reg = riscv.Registers.UNALLOCATED_INT
    start, stop = (create_ssa_value(reg) for _ in range(2))
    op = loop_cls(start, stop, IntegerAttr(1, i12), ())
    op.body.block.add_op(riscv_scf.YieldOp())
    allocator = RegisterAllocatorLivenessBlockNaive(
        RiscvRegisterStack(allow_infinite=True)
    )
    allocator.live_ins_per_block = live_ins_per_block(op.body.block)
    op.allocate_registers(allocator)
    op.verify()

    start, stop = op.start, op.stop
    # With no uses in the body, the initial bound can reuse the IV's register.
    # The termination bound must survive each iteration in a separate register.
    assert start.type == op.body.block.args[0].type
    assert start.type != stop.type


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
    start = create_ssa_value(reg)
    stop = create_ssa_value(reg)
    step_val = create_ssa_value(reg)

    op = loop_cls(
        start,
        stop,
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


def test_riscv_scf_for_update_liveness_not_implemented():
    start = create_ssa_value(riscv.Registers.A0)
    stop = create_ssa_value(riscv.Registers.A1)
    op = riscv_scf.ForOp(start, stop, IntegerAttr(1, i12), ())
    ctx = VerifyLivenessContext(set())
    with pytest.raises(
        NotImplementedError, match="does not yet implement update_liveness"
    ):
        ctx.process_op(op)
