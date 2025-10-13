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
from xdsl.ir import Block, Operation, SSAValue

_ValueRange = Sequence[SSAValue]


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
    successor_inputs: dict[Operation, _ValueRange]

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
        self, predecessor: Operation, inputs: _ValueRange | None = None
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
