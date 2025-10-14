from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Self

from xdsl.analysis.dataflow import (
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,
    ProgramPoint,
)
from xdsl.analysis.sparse_analysis import (
    PropagatingLattice,
)
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import IntegerType, i32
from xdsl.ir import Block, Operation, SSAValue
from xdsl.utils.test_value import create_ssa_value

# region Test lattice implementations


@dataclass(frozen=True)
class SimpleLatticeValue:
    """A simple lattice value for testing AbstractLatticeValue protocol."""

    value: int

    def meet(self, other: SimpleLatticeValue) -> SimpleLatticeValue:
        return SimpleLatticeValue(min(self.value, other.value))

    def join(self, other: SimpleLatticeValue) -> SimpleLatticeValue:
        return SimpleLatticeValue(max(self.value, other.value))


class SimpleSparseLattice(PropagatingLattice):
    """Concrete implementation of SparseLatticeSubscriberBase for testing."""

    def __init__(self, anchor: SSAValue):
        super().__init__(anchor)
        self.value = 0

    def join(self, other: Self) -> ChangeResult:
        if other.value > self.value:
            self.value = other.value
            return ChangeResult.CHANGE
        return ChangeResult.NO_CHANGE

    def meet(self, other: Self) -> ChangeResult:
        if other.value < self.value:
            self.value = other.value
            return ChangeResult.CHANGE
        return ChangeResult.NO_CHANGE

    def __str__(self) -> str:
        return f"SimpleSparseLattice({self.value})"


# endregion


# region AbstractLatticeValue protocol tests


def test_abstract_lattice_value_meet():
    """Test that meet operation works on lattice values."""
    val1 = SimpleLatticeValue(5)
    val2 = SimpleLatticeValue(3)
    result = val1.meet(val2)
    assert result.value == 3


def test_abstract_lattice_value_join():
    """Test that join operation works on lattice values."""
    val1 = SimpleLatticeValue(5)
    val2 = SimpleLatticeValue(3)
    result = val1.join(val2)
    assert result.value == 5


# endregion


# region SparseLatticeSubscriberBase tests


def test_sparse_lattice_initialization():
    """Test that SparseLatticeSubscriberBase initializes correctly."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = SimpleSparseLattice(anchor)

    assert lattice.anchor is anchor
    assert len(lattice.use_def_subscribers) == 0
    assert len(lattice.dependents) == 0


def test_use_def_subscribe():
    """Test that analyses can subscribe to use-def updates."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = SimpleSparseLattice(anchor)

    solver = DataFlowSolver(Context())

    class DummyAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            pass

        def visit(self, point: ProgramPoint) -> None:
            pass

    analysis1 = DummyAnalysis(solver)
    analysis2 = DummyAnalysis(solver)

    lattice.use_def_subscribe(analysis1)
    assert analysis1 in lattice.use_def_subscribers
    assert len(lattice.use_def_subscribers) == 1

    lattice.use_def_subscribe(analysis2)
    assert analysis2 in lattice.use_def_subscribers
    assert len(lattice.use_def_subscribers) == 2

    # Subscribing the same analysis twice shouldn't duplicate
    lattice.use_def_subscribe(analysis1)
    assert len(lattice.use_def_subscribers) == 2


def test_on_update_propagates_to_users():
    """Test that on_update enqueues work for users of the SSA value."""
    # Create operations with use-def chain
    producer = test.TestOp(result_types=[i32])
    consumer1 = test.TestOp(operands=[producer.results[0]])
    consumer2 = test.TestOp(operands=[producer.results[0]])

    Block([producer, consumer1, consumer2])

    # Set up analysis and solver
    solver = DataFlowSolver(Context())
    visited_points: list[ProgramPoint] = []

    class TrackingAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            # Create lattice and subscribe analysis
            lattice = SimpleSparseLattice(producer.results[0])
            lattice.use_def_subscribe(self)
            # Simulate a change to trigger the update
            lattice.on_update(solver)

        def visit(self, point: ProgramPoint) -> None:
            visited_points.append(point)

    solver.load(TrackingAnalysis)

    # Run solver
    solver.initialize_and_run(producer)

    # Check that both users were visited
    assert len(visited_points) == 2
    assert any(p.op is consumer1 for p in visited_points)
    assert any(p.op is consumer2 for p in visited_points)


def test_on_update_calls_super():
    """Test that on_update also propagates to explicit dependents."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = SimpleSparseLattice(anchor)

    solver = DataFlowSolver(Context())
    visited_points: list[ProgramPoint] = []

    class TrackingAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            # Add explicit dependent
            dependent_op = test.TestOp()
            dependent_point = ProgramPoint.before(dependent_op)
            lattice.dependents.add((dependent_point, self))
            # Simulate a change to trigger the update
            lattice.on_update(solver)

        def visit(self, point: ProgramPoint) -> None:
            visited_points.append(point)

    solver.load(TrackingAnalysis)

    # Create a dummy operation to initialize with
    dummy_op = test.TestOp()

    # Run solver
    solver.initialize_and_run(dummy_op)

    # Check that explicit dependent was visited
    assert len(visited_points) == 1


def test_on_update_with_non_ssa_anchor():
    """Test that on_update handles non-SSAValue anchors gracefully."""
    # Create a dummy SSAValue for the test since SimpleSparseLattice requires SSAValue
    anchor = create_ssa_value(IntegerType(32))
    lattice = SimpleSparseLattice(anchor)

    solver = DataFlowSolver(Context())

    class DummyAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            pass

        def visit(self, point: ProgramPoint) -> None:
            pass

    analysis = DummyAnalysis(solver)
    lattice.use_def_subscribe(analysis)

    # Create a dummy operation to initialize with
    dummy_op = test.TestOp()

    # Should not raise an error
    solver.initialize_and_run(dummy_op)


def test_sparse_lattice_join_and_meet():
    """Test that join and meet operations work and return correct ChangeResult."""
    anchor = create_ssa_value(IntegerType(32))
    lattice1 = SimpleSparseLattice(anchor)
    lattice2 = SimpleSparseLattice(anchor)

    lattice1.value = 5
    lattice2.value = 3

    # Join should take maximum and report change
    result = lattice1.join(lattice2)
    assert result == ChangeResult.NO_CHANGE  # 5 >= 3
    assert lattice1.value == 5

    lattice2.value = 10
    result = lattice1.join(lattice2)
    assert result == ChangeResult.CHANGE  # 5 < 10
    assert lattice1.value == 10

    # Meet should take minimum and report change
    lattice1.value = 5
    lattice2.value = 3
    result = lattice1.meet(lattice2)
    assert result == ChangeResult.CHANGE  # 5 > 3
    assert lattice1.value == 3

    result = lattice1.meet(lattice2)
    assert result == ChangeResult.NO_CHANGE  # 3 == 3
    assert lattice1.value == 3


# endregion
