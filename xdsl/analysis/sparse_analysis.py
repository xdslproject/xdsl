from __future__ import annotations

from abc import ABC
from typing import Protocol

from typing_extensions import Self

from xdsl.analysis.dataflow import (
    AnalysisState,
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,  # For type annotations
    ProgramPoint,
)
from xdsl.ir import SSAValue


class AbstractLatticeValue(Protocol):
    """Protocol for types that have join and meet methods."""

    def meet(self, other: Self) -> Self:
        raise NotImplementedError()

    def join(self, other: Self) -> Self:
        raise NotImplementedError()


class AbstractSparseLattice(Protocol):
    """Protocol for types that have join and meet methods."""

    def join(self, other: Self) -> ChangeResult: ...

    def meet(self, other: Self) -> ChangeResult: ...


# TODO: find better name for this
class SparseLatticeSubscriberBase(AnalysisState, AbstractSparseLattice, ABC):
    """
    The base class for a lattice element in a sparse analysis.
    It is attached to an SSAValue.
    """

    use_def_subscribers: set[DataFlowAnalysis]

    def __init__(self, anchor: SSAValue):
        super().__init__(anchor)
        self.use_def_subscribers = set()

    def on_update(self, solver: DataFlowSolver) -> None:
        """
        When a sparse lattice changes, we propagate the change to explicit
        dependents and also to all users of the underlying SSA value for
        subscribed analyses.
        """
        super().on_update(solver)

        if not isinstance(self.anchor, SSAValue):
            return
        for use in self.anchor.uses:
            user_op = use.operation
            user_point = ProgramPoint.before(user_op)
            for analysis in self.use_def_subscribers:
                solver.enqueue((user_point, analysis))

    def use_def_subscribe(self, analysis: DataFlowAnalysis) -> None:
        """
        Subscribe an analysis to be re-invoked on all users of this value
        whenever this lattice state changes.
        """
        self.use_def_subscribers.add(analysis)
