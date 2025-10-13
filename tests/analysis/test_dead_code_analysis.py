from __future__ import annotations

from unittest.mock import Mock

import pytest

from xdsl.analysis.dataflow import (
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,
    LatticeAnchor,
    ProgramPoint,
)
from xdsl.analysis.dead_code_analysis import Executable
from xdsl.dialects import test
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Block, Operation
from xdsl.utils.test_value import create_ssa_value


def test_executable_initial_state():
    """Test the initial state of the Executable lattice element."""
    anchor = Mock(spec=LatticeAnchor)
    state = Executable(anchor)
    assert not state.live
    assert str(state) == "dead"
    assert not state.block_content_subscribers


def test_executable_set_to_live():
    """Test the set_to_live method."""
    anchor = Mock(spec=LatticeAnchor)
    state = Executable(anchor)

    # First time, should change
    assert state.set_to_live() is ChangeResult.CHANGE
    assert state.live
    assert str(state) == "live"

    # Second time, should not change
    assert state.set_to_live() is ChangeResult.NO_CHANGE
    assert state.live
    assert str(state) == "live"


@pytest.fixture
def op_in_block() -> tuple[Operation, Block]:
    op = test.TestOp()
    block = Block([op])
    return op, block


def test_executable_on_update_not_live(op_in_block: tuple[Operation, Block]):
    """Test that on_update does nothing if the state is not live."""
    op, _ = op_in_block
    solver = Mock(spec=DataFlowSolver)
    anchor = ProgramPoint.before(op)
    state = Executable(anchor)
    state.on_update(solver)
    solver.enqueue.assert_not_called()


def test_executable_on_update_not_program_point():
    """Test that on_update does nothing if the anchor is not a ProgramPoint."""
    solver = Mock(spec=DataFlowSolver)
    anchor = create_ssa_value(IntegerType(32))  # Not a ProgramPoint
    state = Executable(anchor)
    state.set_to_live()
    state.on_update(solver)
    solver.enqueue.assert_not_called()


def test_executable_on_update_not_start_of_block(op_in_block: tuple[Operation, Block]):
    """
    Test that on_update does nothing if the program point is not the start of a block.
    """
    op, block = op_in_block
    op2 = test.TestOp()
    block.add_op(op2)

    solver = Mock(spec=DataFlowSolver)
    # Program point after the first op, so not at the start of the block
    anchor = ProgramPoint.after(op)
    state = Executable(anchor)
    state.set_to_live()

    state.on_update(solver)
    solver.enqueue.assert_not_called()


def test_executable_on_update_start_of_block_no_subscribers(
    op_in_block: tuple[Operation, Block],
):
    """Test on_update at the start of a block with no subscribers."""
    op, _ = op_in_block
    solver = Mock(spec=DataFlowSolver)
    anchor = ProgramPoint.before(op)
    state = Executable(anchor)
    state.set_to_live()

    state.on_update(solver)
    solver.enqueue.assert_not_called()


def test_executable_on_update_start_of_block_with_subscribers():
    """Test on_update at the start of a block with subscribers."""
    # Setup a block with multiple ops
    op1 = test.TestOp()
    op2 = test.TestOp()
    _block = Block([op1, op2])

    solver = Mock(spec=DataFlowSolver)
    analysis1 = Mock(spec=DataFlowAnalysis)
    analysis2 = Mock(spec=DataFlowAnalysis)

    anchor = ProgramPoint.before(op1)
    state = Executable(anchor)
    state.set_to_live()
    state.block_content_subscribers.add(analysis1)
    state.block_content_subscribers.add(analysis2)

    state.on_update(solver)

    # Check that all ops are enqueued for all subscribed analyses
    assert solver.enqueue.call_count == 4
    solver.enqueue.assert_any_call((ProgramPoint.before(op1), analysis1))
    solver.enqueue.assert_any_call((ProgramPoint.before(op1), analysis2))
    solver.enqueue.assert_any_call((ProgramPoint.before(op2), analysis1))
    solver.enqueue.assert_any_call((ProgramPoint.before(op2), analysis2))


def test_executable_on_update_empty_block():
    """Test on_update for a live empty block."""
    block = Block()
    solver = Mock(spec=DataFlowSolver)
    analysis = Mock(spec=DataFlowAnalysis)

    # The anchor for an empty block is the block itself
    anchor = ProgramPoint.at_start_of_block(block)
    state = Executable(anchor)
    state.set_to_live()
    state.block_content_subscribers.add(analysis)

    # on_update checks if the anchor's entity is an Operation. For an empty
    # block, it's the block itself, so this should do nothing.
    state.on_update(solver)
    solver.enqueue.assert_not_called()
