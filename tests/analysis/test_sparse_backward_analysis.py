from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from typing_extensions import Self

from xdsl.analysis.dataflow import (
    DataFlowSolver,
    ProgramPoint,
)
from xdsl.analysis.dead_code_analysis import Executable
from xdsl.analysis.sparse_analysis import (
    AbstractLatticeValue,
    Lattice,
    SparseBackwardDataFlowAnalysis,
)
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Block, Operation, Region

# region Test lattice value


@dataclass(frozen=True)
class LiveValue(AbstractLatticeValue):
    """Boolean liveness lattice: live (⊤) ⊑ dead (⊥)."""

    state: Literal["dead", "live"]

    @classmethod
    def initial_value(cls) -> Self:
        return cls("dead")

    def meet(self, other: LiveValue) -> LiveValue:
        # OR semantics: a value is live if either side says so.
        if self.state == "live" or other.state == "live":
            return LiveValue("live")
        return LiveValue("dead")

    def join(self, other: LiveValue) -> LiveValue:
        # AND semantics: dual to meet.
        if self.state == "dead" or other.state == "dead":
            return LiveValue("dead")
        return LiveValue("live")


class LiveLattice(Lattice[LiveValue]):
    value_cls = LiveValue


# endregion


# region Concrete backward analysis


class CopyBackwardAnalysis(SparseBackwardDataFlowAnalysis[LiveLattice]):
    """
    Test analysis: meet each operand with each result. Effectively, an operand
    is live iff any of the op's results are live. No interface dispatch needed.
    """

    def __init__(self, solver: DataFlowSolver):
        super().__init__(solver, LiveLattice)
        self.visited_ops: list[Operation] = []

    def visit_operation_impl(
        self,
        op: Operation,
        operand_lattices: list[LiveLattice],
        result_lattices: list[LiveLattice],
    ) -> None:
        self.visited_ops.append(op)
        for operand in operand_lattices:
            for result in result_lattices:
                self.meet(operand, result)

    def set_to_exit_state(self, lattice: LiveLattice) -> None:
        lattice._value = LiveValue("live")  # pyright: ignore[reportPrivateUsage]


# endregion


def _make_block_executable(solver: DataFlowSolver, block: Block) -> None:
    point = ProgramPoint.at_start_of_block(block)
    solver.get_or_create_state(point, Executable).live = True


# region Initialization tests


def test_backward_analysis_initialization_visits_ops_with_operands():
    """
    Initialization visits every op that has operands. Source ops (no
    operands) are skipped — there is nothing for backward propagation to
    write into them."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = solver.load(CopyBackwardAnalysis)

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(operands=[a.results[0]], result_types=[i32])
    c = test.TestOp(operands=[b.results[0]], result_types=[i32])
    block = Block([a, b, c])
    region = Region([block])
    module = ModuleOp(region)

    _make_block_executable(solver, block)

    solver.initialize_and_run(module)

    assert a not in analysis.visited_ops  # no operands
    assert b in analysis.visited_ops
    assert c in analysis.visited_ops


def test_backward_analysis_initialization_subscribes_block_executable():
    """Initialization subscribes the analysis to block executability."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = solver.load(CopyBackwardAnalysis)

    op = test.TestOp(result_types=[i32])
    block = Block([op])
    region = Region([block])
    module = ModuleOp(region)

    # Block deliberately not marked live yet.
    solver.initialize_and_run(module)

    block_start = ProgramPoint.at_start_of_block(block)
    executable = solver.lookup_state(block_start, Executable)
    assert executable is not None
    assert analysis in executable.block_content_subscribers


# endregion


# region Visit dispatch tests


def test_visit_operation_skips_non_executable_blocks():
    """Operations whose parent block isn't live are not visited."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    op = test.TestOp(result_types=[i32])
    block = Block([op])
    point = ProgramPoint.at_start_of_block(block)
    solver.get_or_create_state(point, Executable).live = False

    analysis.visit_operation(op)
    assert op not in analysis.visited_ops


def test_visit_operation_skips_when_parent_block_not_executable():
    """
    An op with operands whose parent block is not marked live is skipped
    before any lattice propagation happens."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]], result_types=[i32])
    block = Block([producer, consumer])
    # Explicitly mark the block as not live.
    point = ProgramPoint.at_start_of_block(block)
    solver.get_or_create_state(point, Executable).live = False

    analysis.visit_operation(consumer)

    assert consumer not in analysis.visited_ops
    # No lattice should have been created for the consumer's result either,
    # since we bailed out before touching any lattices.
    assert solver.lookup_state(consumer.results[0], LiveLattice) is None


def test_visit_with_block_point_is_noop():
    """
    For backward analysis, visiting a program point whose `op is None`
    (i.e., a block-anchored point) must be a no-op: it must not invoke
    `visit_operation` nor create any lattice state."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    # An empty block yields a ProgramPoint whose entity is the Block itself,
    # so `point.op is None`.
    empty_block = Block([])
    block_point = ProgramPoint.at_start_of_block(empty_block)
    assert block_point.op is None
    assert block_point.block is empty_block

    # Also exercise the end-of-block point of a non-empty block, which is
    # likewise anchored to the Block (op is None).
    op = test.TestOp(result_types=[i32])
    nonempty_block = Block([op])
    end_point = ProgramPoint.at_end_of_block(nonempty_block)
    assert end_point.op is None

    analysis.visit(block_point)
    analysis.visit(end_point)

    assert analysis.visited_ops == []
    assert solver.lookup_state(op.results[0], LiveLattice) is None


def test_visit_operation_registers_result_dependencies():
    """Result lattices must register the op's program point as a dependent."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]], result_types=[i32])
    block = Block([producer, consumer])
    _make_block_executable(solver, block)

    solver._is_running = True  # pyright: ignore[reportPrivateUsage]
    analysis.visit_operation(consumer)

    result_lattice = analysis.get_lattice_element(consumer.results[0])
    consumer_point = ProgramPoint.before(consumer)
    assert (consumer_point, analysis) in result_lattice.dependents


def test_visit_operation_with_regions_raises():
    """Currently, operations with regions are unsupported (when they have operands)."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    producer = test.TestOp(result_types=[i32])
    inner_block = Block([])
    region = Region([inner_block])
    op = test.TestOp(operands=[producer.results[0]], regions=[region])
    outer_block = Block([producer, op])
    _make_block_executable(solver, outer_block)

    with pytest.raises(NotImplementedError, match="RegionBranchOpInterface"):
        analysis.visit_operation(op)


def test_visit_operation_with_successors_raises():
    """Currently, operations with block successors are unsupported."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    producer = test.TestOp(result_types=[i32])
    target = Block([])
    op = test.TestTermOp(operands=[producer.results[0]], successors=[target])
    block = Block([producer, op])
    Region([block, target])
    _make_block_executable(solver, block)

    with pytest.raises(NotImplementedError, match="BranchOpInterface"):
        analysis.visit_operation(op)


# endregion


# region End-to-end propagation tests


def test_backward_propagation_through_chain():
    """Marking the tail of a use-def chain live propagates back to the head."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(CopyBackwardAnalysis)

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(operands=[a.results[0]], result_types=[i32])
    c = test.TestOp(operands=[b.results[0]], result_types=[i32])
    block = Block([a, b, c])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    # Seed: c's result is live. (In a real analysis, this would come from a
    # public-function exit or a side-effect handler.)
    c_result = c.results[0]
    seed = solver.get_or_create_state(c_result, LiveLattice)
    seed._value = LiveValue("live")  # pyright: ignore[reportPrivateUsage]

    solver.initialize_and_run(module)

    # Liveness must have flowed: c's result -> c's operand (= b's result)
    # -> b's operand (= a's result).
    a_lattice = solver.lookup_state(a.results[0], LiveLattice)
    b_lattice = solver.lookup_state(b.results[0], LiveLattice)
    c_lattice = solver.lookup_state(c.results[0], LiveLattice)

    assert a_lattice is not None
    assert a_lattice.value.state == "live"
    assert b_lattice is not None
    assert b_lattice.value.state == "live"
    assert c_lattice is not None
    assert c_lattice.value.state == "live"


def test_backward_propagation_stops_at_dead_results():
    """No seeded liveness anywhere — everything stays dead."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(CopyBackwardAnalysis)

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(operands=[a.results[0]], result_types=[i32])
    block = Block([a, b])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    solver.initialize_and_run(module)

    a_lattice = solver.lookup_state(a.results[0], LiveLattice)
    b_lattice = solver.lookup_state(b.results[0], LiveLattice)

    assert a_lattice is None or a_lattice.value.state == "dead"
    assert b_lattice is None or b_lattice.value.state == "dead"


def test_backward_propagation_through_fan_in():
    """A live result with multiple operands propagates to all of them."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    solver.load(CopyBackwardAnalysis)

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(result_types=[i32])
    c = test.TestOp(operands=[a.results[0], b.results[0]], result_types=[i32])
    block = Block([a, b, c])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    seed = solver.get_or_create_state(c.results[0], LiveLattice)
    seed._value = LiveValue("live")  # pyright: ignore[reportPrivateUsage]

    solver.initialize_and_run(module)

    a_lattice = solver.lookup_state(a.results[0], LiveLattice)
    b_lattice = solver.lookup_state(b.results[0], LiveLattice)
    assert a_lattice is not None
    assert a_lattice.value.state == "live"
    assert b_lattice is not None
    assert b_lattice.value.state == "live"


def test_backward_propagation_reverse_order_one_pass():
    """
    Visiting in reverse order during initialization should reach a fixed point
    without needing the worklist to fire: each op already sees the seeded
    downstream result when visited.
    """
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = solver.load(CopyBackwardAnalysis)

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(operands=[a.results[0]], result_types=[i32])
    c = test.TestOp(operands=[b.results[0]], result_types=[i32])
    block = Block([a, b, c])
    region = Region([block])
    module = ModuleOp(region)
    _make_block_executable(solver, block)

    # Seed c live BEFORE running.
    seed = solver.get_or_create_state(c.results[0], LiveLattice)
    seed._value = LiveValue("live")  # pyright: ignore[reportPrivateUsage]

    solver.initialize_and_run(module)

    # Each op with operands should be visited exactly once during init.
    # (Source op `a` has no operands and is skipped — see visit_operation.)
    # If reverse-order walks correctly, no additional worklist firings are
    # needed for this acyclic chain.
    assert analysis.visited_ops.count(b) == 1
    assert analysis.visited_ops.count(c) == 1


# endregion


# region Helper-method tests


def test_meet_helper_propagates_change():
    """`meet` helper writes through propagate_if_changed."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]], result_types=[i32])
    Block([producer, consumer])

    solver._is_running = True  # pyright: ignore[reportPrivateUsage]

    lhs = analysis.get_lattice_element(producer.results[0])
    rhs = LiveLattice(producer.results[0], value=LiveValue("live"))

    analysis.meet(lhs, rhs)
    assert lhs.value.state == "live"


def test_set_all_to_exit_states():
    """`set_all_to_exit_states` calls `set_to_exit_state` for each lattice."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = CopyBackwardAnalysis(solver)

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(result_types=[i32])

    lattices = [
        analysis.get_lattice_element(a.results[0]),
        analysis.get_lattice_element(b.results[0]),
    ]
    analysis.set_all_to_exit_states(lattices)

    for lattice in lattices:
        assert lattice.value.state == "live"


# endregion
