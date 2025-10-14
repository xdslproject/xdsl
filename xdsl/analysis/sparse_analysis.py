"""
The sparse analysis module provides data flow analysis utilities for sparse lattices.

For more information on lattices, refer to [this Wikipedia article](https://en.wikipedia.org/wiki/Lattice_(order)).

"""

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
    """
    Protocol for the mathematical lattice value types used within Lattice wrappers.
    A lattice is a mathematical structure with a partial ordering and two operations:

    - join (∨): computes the least upper bound (union of information)
    - meet (∧): computes the greatest lower bound (intersection of information)

    Classes implementing this protocol should provide implementations for the `meet` and/or `join` methods.

    This protocol represents the actual lattice element (the abstract value being
    tracked), separate from the propagation infrastructure. For example:

    - In constant propagation: the lattice value might be `⊥ (bottom) | Constant(n) | ⊤ (top)`
    - In sign analysis: the lattice value might be `Positive | Negative | Zero | Unknown`
    - In range analysis: the lattice value might be `Interval(min, max)`
    """

    def meet(self, other: Self) -> Self:
        """
        Computes the greatest lower bound (intersection of information) of two lattice values.

        `a.meet(b)` (or `a ∧ b`) produces the most precise value that is less than or equal
        to both `a` and `b` in the lattice ordering. It represents the combination of two
        abstract values where we keep only information guaranteed to hold in both.

        In other words, `meet` refines information by taking their common part.

        Examples:

        - In constant propagation: `Constant(3) ∧ Constant(3) = Constant(3)`,
          but `Constant(3) ∧ Constant(4) = ⊥ (bottom)`
        - In sign analysis: `Positive ∧ NonZero = Positive`
        - In range analysis: `[0, 10] ∧ [5, 15] = [5, 10]`
        """
        raise NotImplementedError()

    def join(self, other: Self) -> Self:
        """
        Computes the least upper bound (union of information) of two lattice values.

        `a.join(b)` (or `a ∨ b`) produces the least precise value that is greater than or
        equal to both `a` and `b` in the lattice ordering. It represents the merging of
        two abstract values where we keep any information that could hold in either.

        In other words, `join` generalizes information by taking their union.

        Examples:

        - In constant propagation: `Constant(3) ∨ Constant(4) = ⊤ (top)`
        - In sign analysis: `Positive ∨ Negative = Unknown`
        - In range analysis: `[0, 10] ∨ [5, 15] = [0, 15]`
        """
        raise NotImplementedError()


class AbstractSparseLattice(Protocol):
    """
    Protocol for sparse lattice elements used in data flow analysis.

    Seem [AbstractLatticeValue][xdsl.analysis.sparse_analysis.AbstractLatticeValue] for more
    information about lattices and their operations.

    In contrast to [AbstractLatticeValue][xdsl.analysis.sparse_analysis.AbstractLatticeValue],
    the `meet` and `join` methods in this protocol are required to return a ChangeResult,
    signaling whether the lattice element has changed.
    """

    def join(self, other: Self) -> ChangeResult:
        """
        Join two lattice elements. Returns `ChangeResult.CHANGE` if the lattice element has changed,
        otherwise `ChangeResult.NO_CHANGE`.

        For more information about the join operation, see
        [`AbstractLatticeValue.join`][xdsl.analysis.sparse_analysis.AbstractLatticeValue.join].
        """
        ...

    def meet(self, other: Self) -> ChangeResult:
        """
        Meet two lattice elements. Returns `ChangeResult.CHANGE` if the lattice element has changed,
        otherwise `ChangeResult.NO_CHANGE`.

        For more information about the meet operation, see
        [`AbstractLatticeValue.meet`][xdsl.analysis.sparse_analysis.AbstractLatticeValue.meet].
        """
        ...


class PropagatingLattice(AnalysisState, AbstractSparseLattice, ABC):
    """
    Base class for sparse lattice elements attached to SSA values.

    This class implements the infrastructure for propagating lattice changes through
    the data flow analysis framework. When a lattice element changes (e.g., a value
    becomes a known constant), this class ensures that:

    1. All operations that use this SSA value are re-analyzed
    2. Subscribed analyses are notified of the change
    3. The solver's work queue is updated appropriately

    The propagation follows use-def chains: when a lattice attached to an SSA value
    changes, all operations that use that value are marked for re-visiting by any
    analyses that have subscribed to this lattice.

    Subclasses must implement the lattice operations (join/meet) and can override
    on_update() to customize propagation behavior beyond simple use-def chains.

    For the concept of lattices in data flow analysis, see
    [`PropagatingLattice`][xdsl.analysis.sparse_analysis.PropagatingLattice].

    Use this as a base class when you need custom propagation logic (e.g., tracking
    equivalence classes, pointer aliases, or context-sensitive information). For
    simple cases, use the Lattice wrapper instead.
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
