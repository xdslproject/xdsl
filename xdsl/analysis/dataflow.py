"""
Core datastructures and solver for dataflow analyses.
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeAlias

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

AnalysisStateInvT = TypeVar("AnalysisStateInvT", bound="AnalysisState")
DataFlowAnalysisInvT = TypeVar("DataFlowAnalysisInvT", bound="DataFlowAnalysis")


class AnalysisState(ABC, Generic[AnchorInvT]):
    """
    Base class for all analysis states. States are attached to lattice anchors
    and evolve as the analysis iterates.
    """

    anchor: AnchorInvT
    dependents: set[tuple[ProgramPoint, DataFlowAnalysis]]

    def __init__(self, anchor: AnchorInvT):
        self.anchor = anchor
        self.dependents = set()

    def on_update(self, solver: DataFlowSolver) -> None:
        """
        Called by the solver when the state is updated. Enqueues dependent
        work items.
        """
        for point, analysis in self.dependents:
            solver.enqueue((point, analysis))

    @abstractmethod
    def __str__(self) -> str: ...


class DataFlowSolver:
    """
    The main dataflow analysis solver. It orchestrates child analyses, runs
    the fixed-point iteration, and manages analysis states.
    """

    context: Context
    _worklist: collections.deque[tuple[ProgramPoint, DataFlowAnalysis]]
    _is_running: bool

    def __init__(self, context: Context) -> None:
        self.context = context
        self._worklist = collections.deque()

    def enqueue(self, item: tuple[ProgramPoint, DataFlowAnalysis]) -> None:
        """Adds a work item to the solver's worklist."""
        if not self._is_running:
            raise RuntimeError(
                "Cannot enqueue work items when the solver is not running."
            )
        self._worklist.append(item)


class DataFlowAnalysis(ABC):
    """
    Base class for all dataflow analyses. An analysis implements transfer
    functions for IR constructs and is orchestrated by the DataFlowSolver.
    """
