from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.analysis.dataflow import (
    AnalysisState,
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,
    GenericLatticeAnchor,
    LatticeAnchor,
    ProgramPoint,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Operation, SSAValue


@dataclass(frozen=True)
class CFGEdge(GenericLatticeAnchor):
    """A lattice anchor representing a control-flow edge between two blocks."""

    from_block: Block
    to_block: Block

    def __str__(self) -> str:
        from_name = self.from_block.name_hint or id(self.from_block)
        to_name = self.to_block.name_hint or id(self.to_block)
        return f"edge({from_name} -> {to_name})"


class Executable(AnalysisState):
    """A state that represents whether a block or CFG edge is live."""

    live: bool
    block_content_subscribers: set[DataFlowAnalysis]

    def __init__(self, anchor: LatticeAnchor):
        super().__init__(anchor)
        self.live = False
        self.block_content_subscribers = set()

    def set_to_live(self) -> ChangeResult:
        """Marks the anchor as live and returns whether the state changed."""
        if self.live:
            return ChangeResult.NO_CHANGE
        self.live = True
        return ChangeResult.CHANGE

    def on_update(self, solver: DataFlowSolver) -> None:
        super().on_update(solver)

        # If a block becomes live, enqueue its operations for subscribed analyses.
        if self.live and isinstance(self.anchor, ProgramPoint):
            point = self.anchor
            # Check if this is the start of a block
            if (
                isinstance(point.entity, Operation)
                and point.entity.parent
                and point.entity is point.entity.parent.first_op
            ):
                block = point.entity.parent
                for analysis in self.block_content_subscribers:
                    for op in block.ops:
                        solver.enqueue((ProgramPoint.before(op), analysis))

    def __str__(self) -> str:
        return "live" if self.live else "dead"


class PredecessorState(AnalysisState):
    """
    A state representing the set of live control-flow predecessors of a
    program point (e.g., a region entry or a call operation).
    """

    all_known: bool
    known_predecessors: set[Operation]
    successor_inputs: dict[Operation, Sequence[SSAValue]]

    def __init__(self, anchor: LatticeAnchor):
        super().__init__(anchor)
        self.all_known = True
        self.known_predecessors = set()
        self.successor_inputs = {}

    def set_has_unknown_predecessors(self) -> ChangeResult:
        """Marks that not all predecessors can be known."""
        if not self.all_known:
            return ChangeResult.NO_CHANGE
        self.all_known = False
        return ChangeResult.CHANGE

    def join(
        self, predecessor: Operation, inputs: Sequence[SSAValue] | None = None
    ) -> ChangeResult:
        """Add a known predecessor and its successor inputs."""
        changed = predecessor not in self.known_predecessors
        if changed:
            self.known_predecessors.add(predecessor)

        if inputs is not None:
            if self.successor_inputs.get(predecessor) != inputs:
                self.successor_inputs[predecessor] = inputs
                changed = True

        return ChangeResult.CHANGE if changed else ChangeResult.NO_CHANGE

    def __str__(self) -> str:
        preds = ", ".join(p.name for p in self.known_predecessors)
        status = "all known" if self.all_known else "unknown predecessors"
        return f"Predecessors: [{preds}] ({status})"


class DeadCodeAnalysis(DataFlowAnalysis):
    """
    This analysis identifies executable code paths. It is a foundational analysis
    that other dataflow analyses can depend on to correctly handle control flow.

    NOTE: Due to the lack of control flow interfaces in xDSL (e.g., BranchOpInterface,
    RegionBranchOpInterface, CallOpInterface), the analysis does not handle control flow at all.
    Essentially, only the entry block of the top-level op's first region is marked as live.
    """

    def initialize(self, op: Operation) -> None:
        # Mark the entry block of the top-level op's first region as live.
        if op.regions and op.regions[0].first_block:
            entry_point = ProgramPoint.at_start_of_block(op.regions[0].first_block)
            executable = self.get_or_create_state(entry_point, Executable)
            self.propagate_if_changed(executable, executable.set_to_live())

    def visit(self, point: ProgramPoint) -> None:
        op = point.op
        if op is None:
            # This analysis only triggers on operations.
            return
        # special cased for the typical case where the analysis is run on a ModuleOp:
        assert isinstance(op, ModuleOp) or not op.regions, (
            "Cannot yet handle operations with regions"
        )

        # If parent block is not live, do nothing.
        parent_block = op.parent
        if parent_block:
            assert parent_block.last_op
            assert not parent_block.last_op.successors, (
                "Block has successor blocks, which are not supported yet"
            )
            block_start_point = ProgramPoint.at_start_of_block(parent_block)
            parent_executable = self.get_or_create_state(block_start_point, Executable)
            # Create dependency: if block liveness changes, re-visit this op.
            self.add_dependency(parent_executable, point)
            if not parent_executable.live:
                return
            # Subscribe to block liveness to visit all ops if it becomes live.
            parent_executable.block_content_subscribers.add(self)

        # Assert that we don't have nested regions or control flow yet
        assert not op.regions, (
            f"Operation {op.name} has nested regions, which are not supported yet"
        )
