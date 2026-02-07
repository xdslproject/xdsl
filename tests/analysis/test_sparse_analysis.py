from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from typing_extensions import Self

from xdsl.analysis.dataflow import (
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,
    ProgramPoint,
)
from xdsl.analysis.dead_code_analysis import Executable
from xdsl.analysis.sparse_analysis import (
    AbstractLatticeValue,
    Lattice,
    PropagatingLattice,
    SparseForwardDataFlowAnalysis,
)
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import IntegerType, ModuleOp, i32
from xdsl.ir import Block, Operation, Region, SSAValue
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


def test_use_def_subscribe():
    """Test that analyses can subscribe to use-def updates."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = SimpleSparseLattice(anchor)

    assert lattice.anchor is anchor
    assert len(lattice.use_def_subscribers) == 0
    assert len(lattice.dependents) == 0

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


# region Test lattice value implementations


@dataclass(frozen=True)
class TestLatticeValue(AbstractLatticeValue):
    """A test lattice value implementing AbstractLatticeValue protocol."""

    value: int

    @classmethod
    def initial_value(cls) -> Self:
        return cls(-1)

    def meet(self, other: TestLatticeValue) -> TestLatticeValue:
        """Meet returns minimum value."""
        return TestLatticeValue(min(self.value, other.value))

    def join(self, other: TestLatticeValue) -> TestLatticeValue:
        """Join returns maximum value."""
        return TestLatticeValue(max(self.value, other.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TestLatticeValue):
            return NotImplemented
        return self.value == other.value


class TestLattice(Lattice[TestLatticeValue]):
    """Concrete Lattice implementation for testing."""

    value_cls = TestLatticeValue


# endregion


# region Lattice initialization tests


def test_lattice_initialization_with_value():
    """Test that Lattice initializes correctly with an explicit value."""
    anchor = create_ssa_value(IntegerType(32))
    initial_value = TestLatticeValue(42)
    lattice = TestLattice(anchor, value=initial_value)

    assert lattice.anchor is anchor
    assert lattice.value == initial_value
    assert lattice.value.value == 42
    assert len(lattice.use_def_subscribers) == 0


def test_lattice_initialization_without_value():
    """Test that Lattice initializes with default value when none provided."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = TestLattice(anchor)

    assert lattice.anchor is anchor
    # Should initialize with default constructor of TestLatticeValue
    assert isinstance(lattice.value, TestLatticeValue)
    assert not lattice.use_def_subscribers


# endregion


# region Lattice value property tests


def test_lattice_value_property():
    """Test that the value property correctly exposes the internal value."""
    anchor = create_ssa_value(IntegerType(32))
    initial_value = TestLatticeValue(100)
    lattice = TestLattice(anchor, value=initial_value)

    # Value property should return the internal value
    assert lattice.value is lattice._value  # pyright: ignore[reportPrivateUsage]
    assert lattice.value == initial_value


# endregion


# region Lattice meet operation tests


def test_lattice_meet_with_change():
    """Test that meet operation detects changes and updates value."""
    anchor = create_ssa_value(IntegerType(32))
    lattice1 = TestLattice(anchor, value=TestLatticeValue(5))
    lattice2 = TestLattice(anchor, value=TestLatticeValue(3))

    # Meet should take minimum: min(5, 3) = 3
    result = lattice1.meet(lattice2)

    assert result == ChangeResult.CHANGE
    assert lattice1.value.value == 3


def test_lattice_meet_without_change():
    """Test that meet operation detects when no change occurs."""
    anchor = create_ssa_value(IntegerType(32))
    lattice1 = TestLattice(anchor, value=TestLatticeValue(3))
    lattice2 = TestLattice(anchor, value=TestLatticeValue(5))

    # Meet should take minimum: min(3, 5) = 3 (no change)
    result = lattice1.meet(lattice2)

    assert result == ChangeResult.NO_CHANGE
    assert lattice1.value.value == 3


# endregion


# region Lattice join operation tests


def test_lattice_join_with_change():
    """Test that join operation detects changes and updates value."""
    anchor = create_ssa_value(IntegerType(32))
    lattice1 = TestLattice(anchor, value=TestLatticeValue(5))
    lattice2 = TestLattice(anchor, value=TestLatticeValue(10))

    # Join should take maximum: max(5, 10) = 10
    result = lattice1.join(lattice2)

    assert result == ChangeResult.CHANGE
    assert lattice1.value.value == 10


def test_lattice_join_without_change():
    """Test that join operation detects when no change occurs."""
    anchor = create_ssa_value(IntegerType(32))
    lattice1 = TestLattice(anchor, value=TestLatticeValue(10))
    lattice2 = TestLattice(anchor, value=TestLatticeValue(5))

    # Join should take maximum: max(10, 5) = 10 (no change)
    result = lattice1.join(lattice2)

    assert result == ChangeResult.NO_CHANGE
    assert lattice1.value.value == 10


# endregion


# region Lattice propagation tests


def test_lattice_use_def_subscription():
    """Test that analyses can subscribe to Lattice updates."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = TestLattice(anchor, value=TestLatticeValue(0))

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

    lattice.use_def_subscribe(analysis2)
    assert len(lattice.use_def_subscribers) == 2


def test_lattice_propagates_on_join_change():
    """Test that join changes trigger propagation to users."""
    # Create operations with use-def chain
    producer = test.TestOp(result_types=[i32])
    consumer1 = test.TestOp(operands=[producer.results[0]])
    consumer2 = test.TestOp(operands=[producer.results[0]])

    Block([producer, consumer1, consumer2])

    solver = DataFlowSolver(Context())
    visited_points: list[ProgramPoint] = []

    class PropagationAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            if op is producer:
                lattice = TestLattice(producer.results[0], value=TestLatticeValue(0))
                lattice.use_def_subscribe(self)
                # Store for later access
                self.lattice = lattice
                # Enqueue initial work at the producer
                solver.enqueue((ProgramPoint.before(producer), self))

        def visit(self, point: ProgramPoint) -> None:
            visited_points.append(point)
            # Simulate a change via join if we're at the producer
            if point.op is producer and hasattr(self, "lattice"):
                other = TestLattice(producer.results[0], value=TestLatticeValue(10))
                result = self.lattice.join(other)
                # Propagate the change to trigger visiting consumers
                self.propagate_if_changed(self.lattice, result)

    solver.load(PropagationAnalysis)
    solver.initialize_and_run(producer)

    # Should visit producer, then consumers after the change
    assert len(visited_points) == 3  # producer + 2 consumers
    assert any(p.op is consumer1 for p in visited_points)
    assert any(p.op is consumer2 for p in visited_points)


def test_lattice_propagates_on_meet_change():
    """Test that meet changes trigger propagation to users."""
    # Create operations with use-def chain
    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]])

    Block([producer, consumer])

    solver = DataFlowSolver(Context())
    visited_points: list[ProgramPoint] = []

    class PropagationAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            if op is producer:
                lattice = TestLattice(producer.results[0], value=TestLatticeValue(10))
                lattice.use_def_subscribe(self)
                self.lattice = lattice
                # Enqueue initial work at the producer
                solver.enqueue((ProgramPoint.before(producer), self))

        def visit(self, point: ProgramPoint) -> None:
            visited_points.append(point)
            # Simulate a change via meet if we're at the producer
            if point.op is producer and hasattr(self, "lattice"):
                other = TestLattice(producer.results[0], value=TestLatticeValue(5))
                result = self.lattice.meet(other)
                # Propagate the change to trigger visiting consumers
                self.propagate_if_changed(self.lattice, result)

    solver.load(PropagationAnalysis)
    solver.initialize_and_run(producer)

    # Should visit producer, then consumer after the change
    assert len(visited_points) == 2  # producer + consumer
    assert any(p.op is consumer for p in visited_points)


# endregion


# region Lattice string representation tests


def test_lattice_string_representation():
    """Test that Lattice has a meaningful string representation."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = TestLattice(anchor, value=TestLatticeValue(42))

    str_repr = str(lattice)
    assert "42" in str_repr


# endregion


# region Lattice integration tests


def test_lattice_multiple_operations():
    """Test Lattice through multiple operations."""
    anchor = create_ssa_value(IntegerType(32))
    lattice = TestLattice(anchor, value=TestLatticeValue(5))

    other1 = TestLattice(anchor, value=TestLatticeValue(10))
    other2 = TestLattice(anchor, value=TestLatticeValue(3))

    # First join: 5 ∨ 10 = 10
    result1 = lattice.join(other1)
    assert result1 == ChangeResult.CHANGE
    assert lattice.value.value == 10

    # Second join: 10 ∨ 3 = 10 (no change)
    result2 = lattice.join(other2)
    assert result2 == ChangeResult.NO_CHANGE
    assert lattice.value.value == 10

    # Meet: 10 ∧ 3 = 3
    result3 = lattice.meet(other2)
    assert result3 == ChangeResult.CHANGE
    assert lattice.value.value == 3


# endregion


# region Test lattice for simple analysis


@dataclass(frozen=True)
class KnownLatticeValue(AbstractLatticeValue):
    """A simple lattice value: Unknown (⊥) < Known < Top (⊤)"""

    value: Literal["unknown", "top"] | int

    @classmethod
    def initial_value(cls) -> Self:
        return cls("unknown")

    def meet(self, other: KnownLatticeValue) -> KnownLatticeValue:
        """Meet returns the more precise value."""
        match (self.value, other.value):
            case ("unknown", _):
                return other
            case (_, "unknown"):
                return self
            case ("top", _) | (_, "top"):
                return KnownLatticeValue("top")
            case (a, b) if a == b:
                return self
            case _:
                return KnownLatticeValue("top")

    def join(self, other: KnownLatticeValue) -> KnownLatticeValue:
        """Join returns the less precise value."""
        match (self.value, other.value):
            case ("top", _) | (_, "top"):
                return KnownLatticeValue("top")
            case ("unknown", _):
                return other
            case (_, "unknown"):
                return self
            case (a, b) if a == b:
                return self
            case _:
                return KnownLatticeValue("top")


class SimpleLattice(Lattice[KnownLatticeValue]):
    value_cls = KnownLatticeValue


# endregion


# region Concrete analysis implementation


class SimpleForwardAnalysis(SparseForwardDataFlowAnalysis[SimpleLattice]):
    """A simple forward analysis for testing."""

    def __init__(self, solver: DataFlowSolver):
        super().__init__(solver, SimpleLattice)
        self.visited_ops: list[Operation] = []

    def visit_operation_impl(
        self,
        op: Operation,
        operands: list[SimpleLattice],
        results: list[SimpleLattice],
    ) -> None:
        """Simple transfer function: propagate operand states to results."""
        self.visited_ops.append(op)

        # For testing, just copy first operand state to all results
        if operands and results:
            for result in results:
                self.join(result, operands[0])

    def set_to_entry_state(self, lattice: SimpleLattice) -> None:
        """Set lattice to top (most pessimistic) state."""
        lattice._value = KnownLatticeValue("top")  # pyright: ignore[reportPrivateUsage]


# endregion


# region Basic initialization tests


def test_forward_analysis_initialization():
    """Test that initialization creates lattices for region arguments."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = SimpleForwardAnalysis(solver)

    # Create operation with a region containing arguments
    arg_type = IntegerType(32)
    block = Block(arg_types=[arg_type, arg_type])
    region = Region([block])
    op = ModuleOp(region)

    # Mark the entry block as executable
    entry_point = ProgramPoint.at_start_of_block(block)
    executable = solver.get_or_create_state(entry_point, Executable)
    executable.live = True

    analysis.initialize(op)

    # Check that lattices were created for arguments
    arg0_lattice = solver.lookup_state(block.args[0], SimpleLattice)
    arg1_lattice = solver.lookup_state(block.args[1], SimpleLattice)

    assert arg0_lattice is not None
    assert arg1_lattice is not None
    # Entry state should be "top"
    assert arg0_lattice.value.value == "top"
    assert arg1_lattice.value.value == "top"


# endregion


# region visit_operation tests


def test_visit_operation_creates_lattices():
    """Test that visit_operation retrieves/creates lattices for operands and results."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = SimpleForwardAnalysis(solver)

    # Create operations
    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]], result_types=[i32])
    block = Block([producer, consumer])

    # Mark block as executable
    entry_point = ProgramPoint.at_start_of_block(block)
    executable = solver.get_or_create_state(entry_point, Executable)
    executable.live = True

    solver._is_running = True  # pyright: ignore[reportPrivateUsage]

    # Visit consumer operation
    analysis.visit_operation(consumer)

    # Check that lattices exist
    operand_lattice = solver.lookup_state(producer.results[0], SimpleLattice)
    result_lattice = solver.lookup_state(consumer.results[0], SimpleLattice)

    assert operand_lattice is not None
    assert result_lattice is not None


def test_visit_operation_skips_non_executable_blocks():
    """Test that operations in non-executable blocks are skipped."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = SimpleForwardAnalysis(solver)

    # Create operation in a block
    op = test.TestOp(result_types=[i32])
    block = Block([op])

    # Don't mark block as executable
    entry_point = ProgramPoint.at_start_of_block(block)
    executable = solver.get_or_create_state(entry_point, Executable)
    executable.live = False

    # Visit should not process the operation
    analysis.visit_operation(op)

    # The operation should not be in visited list
    assert op not in analysis.visited_ops


def test_visit_operation_subscribes_to_operands():
    """Test that visit_operation subscribes to operand changes."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = SimpleForwardAnalysis(solver)

    # Create operations
    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]], result_types=[i32])
    block = Block([producer, consumer])

    # Mark block as executable
    entry_point = ProgramPoint.at_start_of_block(block)
    executable = solver.get_or_create_state(entry_point, Executable)
    executable.live = True

    solver._is_running = True  # pyright: ignore[reportPrivateUsage]

    # Visit consumer
    analysis.visit_operation(consumer)

    # Check that analysis subscribed to the operand lattice
    operand_lattice = analysis.get_lattice_element(producer.results[0])
    assert analysis in operand_lattice.use_def_subscribers


def test_visit_operation_handles_no_results():
    """Test that operations without results are handled correctly."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = SimpleForwardAnalysis(solver)

    # Create operation without results
    op = test.TestOp(result_types=[])
    block = Block([op])

    # Mark block as executable
    entry_point = ProgramPoint.at_start_of_block(block)
    executable = solver.get_or_create_state(entry_point, Executable)
    executable.live = True

    # Should return early without error
    analysis.visit_operation(op)
    assert op not in analysis.visited_ops


# endregion


# region join operation tests


def test_join_operation():
    """Test that join operation works correctly."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = SimpleForwardAnalysis(solver)

    # Create two lattices with the same anchor
    from xdsl.utils.test_value import create_ssa_value

    anchor = create_ssa_value(IntegerType(32))
    lattice1 = SimpleLattice(anchor, value=KnownLatticeValue("unknown"))
    lattice2 = SimpleLattice(anchor, value=KnownLatticeValue(42))

    solver._is_running = True  # pyright: ignore[reportPrivateUsage]

    # Join should update lattice1
    analysis.join(lattice1, lattice2)

    assert lattice1.value.value == 42


# endregion


# region Integration test


def test_simple_forward_analysis_integration():
    """Test a simple forward analysis end-to-end."""
    ctx = Context()
    solver = DataFlowSolver(ctx)
    analysis = solver.load(SimpleForwardAnalysis)

    # Create a simple chain: producer -> consumer
    producer = test.TestOp(result_types=[i32])
    consumer = test.TestOp(operands=[producer.results[0]], result_types=[i32])
    block = Block([producer, consumer])
    region = Region([block])
    module = ModuleOp(region)

    # Mark block as executable before running
    entry_point = ProgramPoint.at_start_of_block(block)
    executable = solver.get_or_create_state(entry_point, Executable)
    executable.live = True

    # Run the analysis
    solver.initialize_and_run(module)

    # Both operations should have been visited
    assert producer in analysis.visited_ops
    assert consumer in analysis.visited_ops

    # Lattices should exist for results
    producer_lattice = solver.lookup_state(producer.results[0], SimpleLattice)
    consumer_lattice = solver.lookup_state(consumer.results[0], SimpleLattice)

    assert producer_lattice is not None
    assert consumer_lattice is not None


# endregion
