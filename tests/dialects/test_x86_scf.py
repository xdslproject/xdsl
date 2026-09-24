from collections.abc import Callable

import pytest

from xdsl import ir
from xdsl.backend.liveness import VerifyLivenessContext
from xdsl.backend.register_allocator import live_ins_per_block
from xdsl.backend.x86.register_allocation import X86RegisterAllocator
from xdsl.backend.x86.register_stack import X86RegisterStack
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import test, x86, x86_scf
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.x86.ops import si32
from xdsl.traits import (
    EffectInstance,
    MemoryEffect,
    MemoryEffectKind,
    NoMemoryEffect,
    RecursiveMemoryEffect,
    get_effects,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("iter_arg_types", [(), (x86.registers.R11,)])
@pytest.mark.parametrize("gen_block", [True, False])
def test_for_rof_init(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
    iter_arg_types: tuple[ir.Attribute, ...],
    gen_block: bool,
):
    start = create_ssa_value(x86.registers.R12)
    stop = create_ssa_value(x86.registers.R13)
    step = create_ssa_value(x86.registers.R10)
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
    assert len(op.results) == 1 + len(iter_arg_types)
    assert op.iv_end.type == start.type
    with ImplicitBuilder(op.body) as (_i, *args):
        x86_scf.YieldOp(*args)
    op.verify()

    assert op.step is step


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("static_stop", [False, True])
@pytest.mark.parametrize("static_step", [False, True])
def test_for_rof_bounds_and_step(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
    static_stop: bool,
    static_step: bool,
):
    reg = x86.registers.UNALLOCATED_REG64
    start = create_ssa_value(reg)
    stop_val = create_ssa_value(reg)
    step_val = create_ssa_value(reg)
    stop_attr = IntegerAttr(42, si32)
    step_attr = IntegerAttr(3, si32)

    stop = stop_attr if static_stop else stop_val
    step = step_attr if static_step else step_val

    op = loop_cls(start, stop, step, ())
    op.body.block.add_op(x86_scf.YieldOp())
    op.verify()

    assert (op.stop_attr is None) is (not static_stop)
    assert (op.stop_val is None) is static_stop
    assert (op.step_attr is None) is (not static_step)
    assert (op.step_val is None) is static_step

    assert op.stop is stop
    assert op.step is (step_attr if static_step else step_val)
    assert len(op.results) == 1
    assert op.iv_end.type == start.type


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
def test_for_rof_register_constraints(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
):
    start = create_ssa_value(x86.registers.R12)
    stop = create_ssa_value(x86.registers.R13)
    step = create_ssa_value(x86.registers.R10)
    init = create_ssa_value(x86.registers.R11)
    op = loop_cls(start, stop, step, (init,))
    op.body.block.add_op(x86_scf.YieldOp(op.body.block.args[1]))
    op.verify()

    constraints = op.get_register_constraints()
    assert tuple(constraints.ins) == (stop, step)
    assert constraints.outs == ()
    assert constraints.inouts == ((start, op.iv_end), (init, op.res[0]))


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("wrong_result", [False, True])
def test_for_rof_reject_mismatched_iv(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp], wrong_result: bool
):
    start = create_ssa_value(x86.registers.R12)
    stop = create_ssa_value(x86.registers.R13)
    op = loop_cls(start, stop, IntegerAttr(1, si32), ())
    op.body.block.add_op(x86_scf.YieldOp())
    if wrong_result:
        # The result must follow the initial bound, not the termination bound.
        op = loop_cls.create(
            operands=op.operands,
            result_types=[stop.type],
            properties=op.properties,
            regions=[op.detach_region(op.body)],
        )
        message = "Expected induction var to be same type as iv_end result"
    else:
        op.body.block.erase_arg(op.body.block.args[0])
        op.body.block.insert_arg(stop.type, 0)
        message = "Expected induction var to be same type as start"
    with pytest.raises(VerifyException, match=message):
        op.verify()


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
def test_for_rof_allocate_iv(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp],
):
    reg = x86.registers.UNALLOCATED_REG64
    start, stop = (create_ssa_value(reg) for _ in range(2))
    op = loop_cls(start, stop, IntegerAttr(1, si32), ())
    op.body.block.add_op(x86_scf.YieldOp())
    allocator = X86RegisterAllocator(X86RegisterStack(allow_infinite=True))
    allocator.live_ins_per_block = live_ins_per_block(op.body.block)
    op.allocate_registers(allocator)
    op.verify()

    start, stop = op.start, op.stop
    assert isinstance(stop, ir.SSAValue)
    assert op.iv_end.type == op.body.block.args[0].type == start.type
    assert start.type != stop.type


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("live_bound", ["start", "stop"])
def test_for_rof_bound_liveness(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp], live_bound: str
):
    start = create_ssa_value(x86.registers.R12)
    stop = create_ssa_value(x86.registers.R13)
    op = loop_cls(start, stop, IntegerAttr(1, si32), ())
    op.body.block.add_op(x86_scf.YieldOp())
    ctx = VerifyLivenessContext({start if live_bound == "start" else stop})
    if live_bound == "start":
        with pytest.raises(
            VerifyException, match="should not be read after in/out usage"
        ):
            ctx.process_op(op)
    else:
        ctx.process_op(op)


@pytest.mark.parametrize("loop_cls", [x86_scf.ForOp, x86_scf.RofOp])
@pytest.mark.parametrize("alias_step", [False, True])
def test_for_rof_start_aliases_control(
    loop_cls: type[x86_scf.ForOp | x86_scf.RofOp], alias_step: bool
):
    reg = x86.registers.UNALLOCATED_REG64
    start, stop = (create_ssa_value(reg) for _ in range(2))
    step = start if alias_step else create_ssa_value(reg)
    if not alias_step:
        stop = start
    op = loop_cls(start, stop, step, ())
    op.body.block.add_op(x86_scf.YieldOp())
    with pytest.raises(VerifyException, match="should not be read after in/out usage"):
        VerifyLivenessContext(set()).process_op(op)


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
