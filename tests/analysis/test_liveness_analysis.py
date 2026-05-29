from __future__ import annotations

from xdsl.analysis.dataflow import DataFlowSolver, ProgramPoint
from xdsl.analysis.dead_code_analysis import Executable
from xdsl.analysis.liveness_analysis import Liveness, LivenessAnalysis
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Block, Region


def _make_block_executable(solver: DataFlowSolver, block: Block) -> None:
    point = ProgramPoint.at_start_of_block(block)
    solver.get_or_create_state(point, Executable).live = True


def _seed_live(solver: DataFlowSolver, op: test.TestOp | test.TestPureOp) -> None:
    """Seed all results of `op` as live (mirrors a public-function exit)."""
    for r in op.results:
        lattice = solver.get_or_create_state(r, Liveness)
        lattice.is_live = True


# region Memory-effect / side-effecting ops


def test_side_effecting_op_marks_operands_live():
    """An operand of a memory-writing op is live regardless of result use."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    # TestWriteOp has the MemoryWriteEffect trait, so it is NOT trivially dead.
    writer = test.TestWriteOp(operands=[a.results[0]])
    block = Block([a, writer])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    solver.initialize_and_run(module)

    a_liveness = solver.lookup_state(a.results[0], Liveness)
    assert a_liveness is not None
    assert a_liveness.is_live


def test_read_only_op_does_not_force_operands_live():
    """A read-only op is treated as trivially dead; operands stay dead unless
    a downstream live result demands them."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    # TestReadOp only reads, so `would_be_trivially_dead` is True for it.
    reader = test.TestReadOp(operands=[a.results[0]], result_types=[i32])
    block = Block([a, reader])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    solver.initialize_and_run(module)

    a_liveness = solver.lookup_state(a.results[0], Liveness)
    reader_liveness = solver.lookup_state(reader.results[0], Liveness)
    assert a_liveness is None or not a_liveness.is_live
    assert reader_liveness is None or not reader_liveness.is_live


# endregion


# region Pure-op chains


def test_pure_op_with_dead_result_keeps_operands_dead():
    """Pure op with no live result has dead operands."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    b = test.TestPureOp(operands=[a.results[0]], result_types=[i32])
    block = Block([a, b])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    solver.initialize_and_run(module)

    a_liveness = solver.lookup_state(a.results[0], Liveness)
    b_liveness = solver.lookup_state(b.results[0], Liveness)
    assert a_liveness is None or not a_liveness.is_live
    assert b_liveness is None or not b_liveness.is_live


def test_live_result_propagates_to_operand():
    """Seeding a pure op's result as live marks its operand live."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    b = test.TestPureOp(operands=[a.results[0]], result_types=[i32])
    block = Block([a, b])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    _seed_live(solver, b)

    solver.initialize_and_run(module)

    a_liveness = solver.lookup_state(a.results[0], Liveness)
    assert a_liveness is not None
    assert a_liveness.is_live


def test_liveness_propagates_through_pure_chain():
    """A live tail propagates back through a chain of pure ops."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    b = test.TestPureOp(operands=[a.results[0]], result_types=[i32])
    c = test.TestPureOp(operands=[b.results[0]], result_types=[i32])
    block = Block([a, b, c])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    _seed_live(solver, c)

    solver.initialize_and_run(module)

    for value in (a.results[0], b.results[0], c.results[0]):
        liveness = solver.lookup_state(value, Liveness)
        assert liveness is not None
        assert liveness.is_live


def test_side_effect_at_tail_propagates_liveness_back():
    """A real-world style chain: pure producer → pure transform → side-effecting
    consumer. The consumer's side effect alone keeps the whole chain live."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    b = test.TestPureOp(operands=[a.results[0]], result_types=[i32])
    writer = test.TestWriteOp(operands=[b.results[0]])
    block = Block([a, b, writer])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    solver.initialize_and_run(module)

    a_liveness = solver.lookup_state(a.results[0], Liveness)
    b_liveness = solver.lookup_state(b.results[0], Liveness)
    assert a_liveness is not None
    assert a_liveness.is_live
    assert b_liveness is not None
    assert b_liveness.is_live


def test_fan_in_to_live_result_marks_all_operands_live():
    """A single live result with multiple operands marks each operand live."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(LivenessAnalysis)

    a = test.TestPureOp(result_types=[i32])
    b = test.TestPureOp(result_types=[i32])
    c = test.TestPureOp(operands=[a.results[0], b.results[0]], result_types=[i32])
    block = Block([a, b, c])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    _seed_live(solver, c)

    solver.initialize_and_run(module)

    for value in (a.results[0], b.results[0]):
        liveness = solver.lookup_state(value, Liveness)
        assert liveness is not None
        assert liveness.is_live


# endregion


# region Liveness lattice unit tests


def test_liveness_mark_live_returns_change_only_on_transition():
    a = test.TestPureOp(result_types=[i32])
    lattice = Liveness(a.results[0])
    assert not lattice.is_live

    from xdsl.analysis.dataflow import ChangeResult

    assert lattice.mark_live() == ChangeResult.CHANGE
    assert lattice.is_live
    assert lattice.mark_live() == ChangeResult.NO_CHANGE


def test_liveness_meet_is_or():
    from xdsl.analysis.dataflow import ChangeResult

    a = test.TestPureOp(result_types=[i32])
    b = test.TestPureOp(result_types=[i32])
    lhs = Liveness(a.results[0])
    rhs = Liveness(b.results[0])

    # dead meet dead -> dead, no change
    assert lhs.meet(rhs) == ChangeResult.NO_CHANGE
    assert not lhs.is_live

    # live meet dead -> live unchanged
    lhs.is_live = True
    assert lhs.meet(rhs) == ChangeResult.NO_CHANGE
    assert lhs.is_live

    # dead meet live -> live, change
    lhs2 = Liveness(a.results[0])
    rhs.is_live = True
    assert lhs2.meet(rhs) == ChangeResult.CHANGE
    assert lhs2.is_live


def test_liveness_str():
    a = test.TestPureOp(result_types=[i32])
    lattice = Liveness(a.results[0])

    assert str(lattice) == "dead"
    lattice.mark_live()
    assert str(lattice) == "live"


def test_set_to_exit_state_marks_live():
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = LivenessAnalysis(solver)

    a = test.TestPureOp(result_types=[i32])
    lattice = analysis.get_lattice_element(a.results[0])

    solver._is_running = True  # pyright: ignore[reportPrivateUsage]
    analysis.set_to_exit_state(lattice)
    assert lattice.is_live

    # Idempotent — calling again is a no-op (does not raise).
    analysis.set_to_exit_state(lattice)
    assert lattice.is_live


# endregion
