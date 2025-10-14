from __future__ import annotations

from xdsl.analysis.dataflow import (
    AnalysisState,
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,
    LatticeAnchor,
    ProgramPoint,
)
from xdsl.ir import Operation


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
