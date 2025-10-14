from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from xdsl.analysis.dataflow import (
    AnalysisState,
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,
    GenericLatticeAnchor,
    ProgramPoint,
)
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.builtin import IntegerType, UnregisteredOp
from xdsl.ir import Block, Operation
from xdsl.utils.test_value import create_ssa_value


# region ChangeResult tests
def test_change_result_or():
    assert ChangeResult.CHANGE | ChangeResult.CHANGE is ChangeResult.CHANGE
    assert ChangeResult.CHANGE | ChangeResult.NO_CHANGE is ChangeResult.CHANGE
    assert ChangeResult.NO_CHANGE | ChangeResult.CHANGE is ChangeResult.CHANGE
    assert ChangeResult.NO_CHANGE | ChangeResult.NO_CHANGE is ChangeResult.NO_CHANGE


# endregion


# region GenericLatticeAnchor tests
@dataclass(frozen=True)
class MyAnchor(GenericLatticeAnchor):
    name: str

    def __str__(self) -> str:
        return f"MyAnchor({self.name})"


def test_generic_lattice_anchor():
    anchor1 = MyAnchor("a")
    anchor2 = MyAnchor("a")
    anchor3 = MyAnchor("b")

    assert anchor1 == anchor2
    assert anchor1 != anchor3
    assert hash(anchor1) == hash(anchor2)
    assert hash(anchor1) != hash(anchor3)
    assert str(anchor1) == "MyAnchor(a)"
    assert not (anchor1 == "a")


# endregion


# region ProgramPoint tests
@pytest.fixture
def ops_in_block() -> tuple[Operation, Operation, Operation, Block]:
    context = Context()
    context.allow_unregistered = True
    op1_class = context.get_optional_op("test_op1")
    op2_class = context.get_optional_op("test_op2")
    op3_class = context.get_optional_op("test_op3")
    assert op1_class is not None
    assert op2_class is not None
    assert op3_class is not None
    op1 = op1_class()
    op2 = op2_class()
    op3 = op3_class()
    block = Block([op1, op2, op3])
    return op1, op2, op3, block


@pytest.fixture
def empty_block() -> Block:
    return Block()


@pytest.fixture
def detached_op() -> Operation:
    return UnregisteredOp.with_name("test.detached")()


def test_program_point_before(
    ops_in_block: tuple[Operation, Operation, Operation, Block],
):
    _, op2, _, _ = ops_in_block
    pp = ProgramPoint.before(op2)
    assert pp.entity is op2


def test_program_point_after(
    ops_in_block: tuple[Operation, Operation, Operation, Block],
):
    op1, op2, op3, block = ops_in_block

    # After op in middle of block
    pp_after_op1 = ProgramPoint.after(op1)
    assert pp_after_op1.entity is op2

    # After last op in block
    pp_after_op3 = ProgramPoint.after(op3)
    assert pp_after_op3.entity is block


def test_program_point_after_detached(detached_op: Operation):
    with pytest.raises(
        ValueError, match="Cannot get ProgramPoint after a detached operation."
    ):
        ProgramPoint.after(detached_op)


def test_program_point_at_block_boundaries(
    ops_in_block: tuple[Operation, Operation, Operation, Block], empty_block: Block
):
    op1, _, _, block = ops_in_block

    # Start of non-empty block
    pp_start = ProgramPoint.at_start_of_block(block)
    assert pp_start.entity is op1

    # End of non-empty block
    pp_end = ProgramPoint.at_end_of_block(block)
    assert pp_end.entity is block

    # Start of empty block
    pp_start_empty = ProgramPoint.at_start_of_block(empty_block)
    assert pp_start_empty.entity is empty_block

    # End of empty block
    pp_end_empty = ProgramPoint.at_end_of_block(empty_block)
    assert pp_end_empty.entity is empty_block


def test_program_point_properties(
    ops_in_block: tuple[Operation, Operation, Operation, Block],
):
    op1, _, _, block = ops_in_block

    pp_before_op = ProgramPoint.before(op1)
    assert pp_before_op.op is op1
    assert pp_before_op.block is block

    pp_at_end = ProgramPoint.at_end_of_block(block)
    assert pp_at_end.op is None
    assert pp_at_end.block is block


# endregion


# region AnalysisState tests
class MyState(AnalysisState):
    def __str__(self) -> str:
        return "MyState"


def test_analysis_state_on_update():
    solver = Mock(spec=DataFlowSolver)
    anchor = create_ssa_value(IntegerType(32))
    state = MyState(anchor)

    point1 = ProgramPoint.before(test.TestOp())
    analysis1 = Mock(spec=DataFlowAnalysis)

    point2 = ProgramPoint.at_end_of_block(Block())
    analysis2 = Mock(spec=DataFlowAnalysis)

    state.dependents.add((point1, analysis1))
    state.dependents.add((point2, analysis2))

    state.on_update(solver)

    solver.enqueue.assert_any_call((point1, analysis1))
    solver.enqueue.assert_any_call((point2, analysis2))
    assert solver.enqueue.call_count == 2


# endregion


# region DataFlowSolver tests
@pytest.fixture
def solver() -> DataFlowSolver:
    return DataFlowSolver(Context())


class MyAnalysis(DataFlowAnalysis):
    def initialize(self, op: Operation) -> None:
        pass

    def visit(self, point: ProgramPoint) -> None:
        pass


def test_data_flow_solver_init(solver: DataFlowSolver):
    # solver should be able to load analyses
    # and accept basic operations without errors
    assert isinstance(solver.context, Context)
    analysis = solver.load(MyAnalysis)
    assert isinstance(analysis, MyAnalysis)


def test_data_flow_solver_load(solver: DataFlowSolver):
    analysis = solver.load(MyAnalysis)
    assert isinstance(analysis, MyAnalysis)
    assert analysis.solver is solver


def test_data_flow_solver_load_while_running_raises(solver: DataFlowSolver):
    # Create an analysis that tries to load another analysis during execution
    class LoadingAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            # Try to load while running
            with pytest.raises(
                RuntimeError,
                match="Cannot load new analyses while the solver is running.",
            ):
                self.solver.load(MyAnalysis)

        def visit(self, point: ProgramPoint) -> None:
            pass

    solver.load(LoadingAnalysis)
    solver.initialize_and_run(Mock())


def test_data_flow_solver_get_lookup_state(solver: DataFlowSolver):
    anchor = create_ssa_value(IntegerType(32))

    # Test get_or_create_state
    state1 = solver.get_or_create_state(anchor, MyState)
    assert isinstance(state1, MyState)
    assert state1.anchor is anchor

    # Test that it returns the same state
    state2 = solver.get_or_create_state(anchor, MyState)
    assert state1 is state2

    # Test lookup_state
    state3 = solver.lookup_state(anchor, MyState)
    assert state1 is state3

    # Test lookup_state for non-existent state
    anchor2 = create_ssa_value(IntegerType(64))
    assert solver.lookup_state(anchor2, MyState) is None
    assert solver.lookup_state(anchor, AnalysisState) is None


def test_data_flow_solver_enqueue_not_running(solver: DataFlowSolver):
    # This tests the public contract - enqueue should only work while running
    with pytest.raises(
        RuntimeError, match="Cannot enqueue work items when the solver is not running."
    ):
        solver.enqueue((Mock(), Mock()))


def test_data_flow_solver_propagate_not_running(solver: DataFlowSolver):
    # This tests the public contract - propagate should only work while running
    with pytest.raises(
        RuntimeError, match="Cannot propagate changes when the solver is not running."
    ):
        solver.propagate_if_changed(Mock(), ChangeResult.CHANGE)


def test_data_flow_solver_run_twice_raises(solver: DataFlowSolver):
    # Create an analysis that tries to run the solver again during execution
    class ReentrantAnalysis(DataFlowAnalysis):
        def initialize(self, op: Operation) -> None:
            # Try to run again while already running
            with pytest.raises(RuntimeError, match="Solver is already running."):
                self.solver.initialize_and_run(Mock())

        def visit(self, point: ProgramPoint) -> None:
            pass

    solver.load(ReentrantAnalysis)
    solver.initialize_and_run(Mock())


# endregion
