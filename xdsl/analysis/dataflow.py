"""
Core datastructures and solver for dataflow analyses.
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeAlias, cast

from typing_extensions import TypeVar

from xdsl.context import Context
from xdsl.ir import Block, Operation, SSAValue


class ChangeResult(Enum):
    """
    A result type used to indicate if a change happened.
    """

    NO_CHANGE = 0
    CHANGE = 1

    def __or__(self, other: ChangeResult) -> ChangeResult:
        return ChangeResult.CHANGE if self == ChangeResult.CHANGE else other


@dataclass(frozen=True)
class GenericLatticeAnchor(ABC):
    """
    Abstract base class for custom lattice anchors. In dataflow analysis,
    lattices are attached to 'anchors'. These are typically IR constructs
    like SSAValue or ProgramPoint, but can be custom constructs for concepts
    like control-flow edges.
    """

    @abstractmethod
    def __str__(self) -> str: ...


@dataclass(frozen=True)
class ProgramPoint:
    """
    A point in the program, either before an operation or at the end of a block.
    This is used as an anchor for dense analyses.
    """

    entity: Operation | Block

    @staticmethod
    def before(op: Operation) -> ProgramPoint:
        """Returns the program point just before an operation."""
        return ProgramPoint(op)

    @staticmethod
    def after(op: Operation) -> ProgramPoint:
        """Returns the program point just after an operation."""
        if op.next_op is not None:
            return ProgramPoint(op.next_op)
        if op.parent is None:
            raise ValueError("Cannot get ProgramPoint after a detached operation.")
        return ProgramPoint(op.parent)

    @staticmethod
    def at_start_of_block(block: Block) -> ProgramPoint:
        """Returns the program point at the start of a block."""
        if block.first_op is None:
            # If block is empty, the start is the end.
            return ProgramPoint(block)
        return ProgramPoint(block.first_op)

    @staticmethod
    def at_end_of_block(block: Block) -> ProgramPoint:
        """Returns the program point at the end of a block."""
        return ProgramPoint(block)

    @property
    def op(self) -> Operation | None:
        """The operation this point is before, or None if at the end of a block."""
        if isinstance(self.entity, Operation):
            return self.entity
        return None

    @property
    def block(self) -> Block | None:
        """The block this point is in, or the block itself if at the end."""
        if isinstance(self.entity, Operation):
            return self.entity.parent
        return self.entity

    def __str__(self) -> str:
        if isinstance(self.entity, Operation):
            return f"before op '{self.entity.name}'"
        return f"at end of block '{self.entity.name_hint or id(self.entity)}'"


LatticeAnchor: TypeAlias = SSAValue | Block | ProgramPoint | GenericLatticeAnchor
"""Union type for all possible lattice anchors."""

AnchorInvT = TypeVar("AnchorInvT", bound=LatticeAnchor, default=LatticeAnchor)
AnchorCovT = TypeVar(
    "AnchorCovT", bound=LatticeAnchor, default=LatticeAnchor, covariant=True
)

AnalysisStateInvT = TypeVar("AnalysisStateInvT", bound="AnalysisState")
DataFlowAnalysisInvT = TypeVar("DataFlowAnalysisInvT", bound="DataFlowAnalysis")


class AnalysisState(ABC, Generic[AnchorCovT]):
    """
    Base class for all analysis states. States are attached to lattice anchors
    and evolve as the analysis iterates.
    """

    _anchor: AnchorCovT
    dependents: set[tuple[ProgramPoint, DataFlowAnalysis]]

    def __init__(self, anchor: AnchorCovT):
        self._anchor = anchor
        self.dependents = set()

    @property
    def anchor(self) -> AnchorCovT:
        return self._anchor

    def on_update(self, solver: DataFlowSolver) -> None:
        """
        Called by the solver when the state is updated. Enqueues dependent
        work items.
        """
        for point, analysis in self.dependents:
            solver.enqueue((point, analysis))

    @abstractmethod
    def __str__(self) -> str: ...


@dataclass
class DataFlowSolver:
    """
    The main dataflow analysis solver. It orchestrates child analyses, runs
    the fixed-point iteration, and manages analysis states.
    """

    context: Context
    _analyses: list[DataFlowAnalysis] = field(default_factory=list["DataFlowAnalysis"])
    _worklist: collections.deque[tuple[ProgramPoint, DataFlowAnalysis]] = field(
        default_factory=collections.deque[tuple[ProgramPoint, "DataFlowAnalysis"]]
    )
    _analysis_states: dict[LatticeAnchor, dict[type[AnalysisState], AnalysisState]] = (
        field(default_factory=lambda: collections.defaultdict(dict))
    )
    _is_running: bool = field(default=False)

    def load(
        self, analysis_class: type[DataFlowAnalysisInvT], *args: Any
    ) -> DataFlowAnalysisInvT:
        """Registers a new analysis with the solver."""
        if self._is_running:
            raise RuntimeError("Cannot load new analyses while the solver is running.")
        analysis = analysis_class(self, *args)
        self._analyses.append(analysis)
        return analysis

    def initialize_and_run(self, op: Operation) -> None:
        """Initializes all analyses and runs the solver to a fixed point."""
        if self._is_running:
            raise RuntimeError("Solver is already running.")
        self._is_running = True
        try:
            for analysis in self._analyses:
                analysis.initialize(op)

            while self._worklist:
                point, analysis = self._worklist.popleft()
                analysis.visit(point)
        finally:
            self._is_running = False

    def get_or_create_state(
        self, anchor: LatticeAnchor, state_type: type[AnalysisStateInvT]
    ) -> AnalysisStateInvT:
        """
        Get the state for a given anchor. If it doesn't exist, create it.
        """
        if state_type not in self._analysis_states[anchor]:
            self._analysis_states[anchor][state_type] = state_type(anchor)
        return cast(AnalysisStateInvT, self._analysis_states[anchor][state_type])

    def lookup_state(
        self, anchor: LatticeAnchor, state_type: type[AnalysisStateInvT]
    ) -> AnalysisStateInvT | None:
        """Look up an analysis state. Returns None if it doesn't exist."""
        if (
            anchor in self._analysis_states
            and state_type in self._analysis_states[anchor]
        ):
            return cast(AnalysisStateInvT, self._analysis_states[anchor][state_type])
        return None

    def enqueue(self, item: tuple[ProgramPoint, DataFlowAnalysis]) -> None:
        """Adds a work item to the solver's worklist."""
        if not self._is_running:
            raise RuntimeError(
                "Cannot enqueue work items when the solver is not running."
            )
        self._worklist.append(item)

    def propagate_if_changed(self, state: AnalysisState, changed: ChangeResult) -> None:
        """If the state has changed, trigger its `on_update` hook."""
        if not self._is_running:
            raise RuntimeError(
                "Cannot propagate changes when the solver is not running."
            )
        if changed == ChangeResult.CHANGE:
            state.on_update(self)


class DataFlowAnalysis(ABC):
    """
    Base class for all dataflow analyses. An analysis implements transfer
    functions for IR constructs and is orchestrated by the DataFlowSolver.
    """

    solver: DataFlowSolver

    def __init__(self, solver: DataFlowSolver):
        self.solver = solver

    @abstractmethod
    def initialize(self, op: Operation) -> None:
        """
        Initializes the analysis, setting up initial states and dependencies
        for a given top-level operation.
        """
        ...

    @abstractmethod
    def visit(self, point: ProgramPoint) -> None:
        """
        The transfer function for a given program point. This is called by the
        solver when a dependency of this analysis at this point has changed.
        """
        ...

    def get_or_create_state(
        self, anchor: LatticeAnchor, state_type: type[AnalysisStateInvT]
    ) -> AnalysisStateInvT:
        """Helper to get or create a state from the solver."""
        return self.solver.get_or_create_state(anchor, state_type)

    def get_state(
        self, anchor: LatticeAnchor, state_type: type[AnalysisStateInvT]
    ) -> AnalysisStateInvT | None:
        """Helper to look up a state from the solver."""
        return self.solver.lookup_state(anchor, state_type)

    def add_dependency(
        self, state: AnalysisState, dependent_point: ProgramPoint
    ) -> None:
        """
        Adds a dependency: if `state` changes, `self.visit(dependent_point)`
        will be called.
        """
        state.dependents.add((dependent_point, self))

    def propagate_if_changed(self, state: AnalysisState, changed: ChangeResult) -> None:
        """Helper to propagate a state change to the solver."""
        self.solver.propagate_if_changed(state, changed)
