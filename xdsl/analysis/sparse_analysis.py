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

    This protocol represents the actual lattice element (the abstract value being
    tracked), separate from the propagation infrastructure. For example:

    - In constant propagation: the lattice value might be `Bottom | Constant(n) | Top`
    - In sign analysis: the lattice value might be `Positive | Negative | Zero | Unknown`
    - In range analysis: the lattice value might be `Interval(min, max)`
    """

    def meet(self, other: Self) -> Self:
        raise NotImplementedError()

    def join(self, other: Self) -> Self:
        raise NotImplementedError()


class AbstractSparseLattice(Protocol):
    """
    Protocol for sparse lattice elements used in data flow analysis.

    A lattice is a mathematical structure with a partial ordering and two operations:

    - join (∨): computes the least upper bound (union of information)
    - meet (∧): computes the greatest lower bound (intersection of information)

    In data flow analysis, lattices represent abstract values or properties that
    flow through the program. For example, in constant propagation:

    - ⊥ (bottom) means "undefined/no information"
    - specific constants like `5`, `7`, etc.
    - ⊤ (top) means "not a constant/conflicting information"
    """

    def join(self, other: Self) -> ChangeResult: ...

    def meet(self, other: Self) -> ChangeResult: ...


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
