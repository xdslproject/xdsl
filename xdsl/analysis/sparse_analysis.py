"""
The sparse analysis module provides data flow analysis utilities for sparse lattices.

For more information on lattices, refer to [this Wikipedia article](https://en.wikipedia.org/wiki/Lattice_(order)).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol

from typing_extensions import Self, TypeVar

from xdsl.analysis.dataflow import (
    AnalysisState,
    ChangeResult,
    DataFlowAnalysis,
    DataFlowSolver,  # For type annotations
    ProgramPoint,
)
from xdsl.analysis.dead_code_analysis import CFGEdge, Executable
from xdsl.ir import Block, Operation, SSAValue


class AbstractLatticeValue(Protocol):
    """
    Protocol for the mathematical lattice value types used within Lattice wrappers.
    A lattice is a mathematical structure with a partial ordering and two operations:

    - join (∨): computes the least upper bound (union of information)
    - meet (∧): computes the greatest lower bound (intersection of information)

    Classes implementing this protocol should provide implementations for the `meet`
    and/or `join` methods. The class should also define the classmethod `initial_value`
    that takes no additional arguments and returns an initial lattice value.

    This protocol represents the actual lattice element (the abstract value being
    tracked), separate from the propagation infrastructure. For example:

    - In constant propagation: the lattice value might be `⊥ (bottom) | Constant(n) | ⊤ (top)`
    - In sign analysis: the lattice value might be `Positive | Negative | Zero | Unknown`
    - In range analysis: the lattice value might be `Interval(min, max)`
    """

    @classmethod
    def initial_value(cls) -> Self:
        """
        Returns an initial lattice value, typically the
        bottom (⊥) or uninitialized state of the lattice.
        """
        ...

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
        ...

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
        ...


class AbstractSparseLattice(Protocol):
    """
    Protocol for sparse lattice elements used in data flow analysis.

    See [AbstractLatticeValue][xdsl.analysis.sparse_analysis.AbstractLatticeValue] for more
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


class PropagatingLattice(AnalysisState[SSAValue], AbstractSparseLattice, ABC):
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


AbstractLatticeValueInvT = TypeVar(
    "AbstractLatticeValueInvT", bound=AbstractLatticeValue
)


class Lattice(PropagatingLattice, Generic[AbstractLatticeValueInvT]):
    """
    Generic wrapper that combines a lattice value with sparse propagation infrastructure.

    See [`AbstractLatticeValue`][xdsl.analysis.sparse_analysis.AbstractLatticeValue] for more information around
    the meet and join operations that need to be implemented.

    If you need meet/join functionality that does not match with what `Lattice` provides,
    consider using [`PropagatingLattice`][xdsl.analysis.sparse_analysis.PropagatingLattice] directly.
    """

    value_cls: type[AbstractLatticeValueInvT]
    _value: AbstractLatticeValueInvT

    def __init__(
        self,
        anchor: SSAValue,
        value: AbstractLatticeValueInvT | None = None,
    ):
        super().__init__(anchor)
        if value is not None:
            self._value = value
        else:
            self._value = self.value_cls.initial_value()

    @property
    def value(self) -> AbstractLatticeValueInvT:
        return self._value

    def meet(self, other: Self) -> ChangeResult:
        new_value = self.value.meet(other.value)

        if new_value == self.value:
            return ChangeResult.NO_CHANGE

        self._value = new_value
        return ChangeResult.CHANGE

    def join(self, other: Self) -> ChangeResult:
        new_value = self.value.join(other.value)

        if new_value == self.value:
            return ChangeResult.NO_CHANGE

        self._value = new_value
        return ChangeResult.CHANGE

    def __str__(self) -> str:
        return str(self.value)


PropagatingLatticeInvT = TypeVar("PropagatingLatticeInvT", bound=PropagatingLattice)


class SparseForwardDataFlowAnalysis(
    DataFlowAnalysis, ABC, Generic[PropagatingLatticeInvT]
):
    """
    Base class for sparse forward data-flow analyses. It propagates lattices
    attached to SSA values along the direction of data flow.
    """

    def __init__(
        self, solver: DataFlowSolver, lattice_type: type[PropagatingLatticeInvT]
    ):
        super().__init__(solver)
        self.lattice_type = lattice_type

    def initialize(self, op: Operation) -> None:
        # Set entry state for all arguments of the top-level regions.
        for region in op.regions:
            if region.first_block is not None:
                for arg in region.first_block.args:
                    self.set_to_entry_state(self.get_lattice_element(arg))

        # Iteratively visit all ops to build initial dependencies.
        stack = [op]

        while stack:
            current_op = stack.pop()

            self.visit(ProgramPoint.before(current_op))

            for region in current_op.regions:
                for block in region.blocks:
                    block_start_point = ProgramPoint.at_start_of_block(block)
                    executable = self.get_or_create_state(block_start_point, Executable)
                    executable.block_content_subscribers.add(self)
                    self.visit(block_start_point)

                    # Add nested ops to stack in reverse order to maintain traversal order
                    stack.extend(reversed(block.ops))

    def visit(self, point: ProgramPoint) -> None:
        if point.op is not None:
            self.visit_operation(point.op)
        elif point.block is not None:
            # This case handles end-of-block points, which for forward analysis
            # means we are visiting the block itself to handle its arguments.
            self.visit_block(point.block)

    def visit_operation(self, op: Operation) -> None:
        """Transfer function for an operation's results."""
        if not op.results:
            return

        # If the parent block is not executable, do nothing.
        if (
            op.parent is not None
            and not self.get_or_create_state(
                ProgramPoint.at_start_of_block(op.parent), Executable
            ).live
        ):
            return

        # Get operand and result lattices
        point = ProgramPoint.before(op)
        operands = [self.get_lattice_element_for(point, o) for o in op.operands]
        results = [self.get_lattice_element(r) for r in op.results]

        # Subscribe to operand changes for future updates.
        for o in op.operands:
            self.get_lattice_element(o).use_def_subscribe(self)

        # Check if operation requires special interface support
        if op.regions:
            raise NotImplementedError(
                f"Operation {op.name} has regions. Full support requires "
                "RegionBranchOpInterface to properly handle control flow "
                "between regions."
            )

        # TODO: handle call operations when CallOpInterface is implemented

        self.visit_operation_impl(op, operands, results)

    def visit_block(self, block: Block) -> None:
        """Transfer function for a block's arguments."""
        if not block.args:
            return

        point = ProgramPoint.at_start_of_block(block)
        if not self.get_or_create_state(point, Executable).live:
            return

        # For non-entry blocks, join values from predecessors.
        if block.parent is None or block.parent.first_block is not block:
            for pred_block in block.predecessors():
                edge = CFGEdge(pred_block, block)
                executable = self.get_state(edge, Executable)
                if not executable or not executable.live:
                    continue

                terminator = pred_block.last_op
                if terminator is None:
                    continue

                # Requires BranchOpInterface to correctly map terminator operands
                # to successor block arguments
                raise NotImplementedError(
                    f"Mapping values across control flow edges requires "
                    "BranchOpInterface. Cannot determine which operands of "
                    f"terminator {terminator.name} correspond to arguments of "
                    f"successor block."
                )
        # else:  # For entry blocks of regions
        # TODO: Requires CallableOpInterface and RegionBranchOpInterface

    def join(self, lhs: PropagatingLatticeInvT, rhs: PropagatingLatticeInvT) -> None:
        """Joins the rhs lattice into the lhs and propagates if changed."""
        self.propagate_if_changed(lhs, lhs.join(rhs))

    def get_lattice_element(self, value: SSAValue) -> PropagatingLatticeInvT:
        return self.get_or_create_state(value, self.lattice_type)

    def get_lattice_element_for(
        self, point: ProgramPoint, value: SSAValue
    ) -> PropagatingLatticeInvT:
        lattice = self.get_lattice_element(value)
        self.add_dependency(lattice, point)
        return lattice

    def set_all_to_entry_state(self, lattices: list[PropagatingLatticeInvT]) -> None:
        for lattice in lattices:
            self.set_to_entry_state(lattice)

    @abstractmethod
    def visit_operation_impl(
        self,
        op: Operation,
        operands: list[PropagatingLatticeInvT],
        results: list[PropagatingLatticeInvT],
    ) -> None:
        """The user-defined transfer function for a generic operation."""
        ...

    @abstractmethod
    def set_to_entry_state(self, lattice: PropagatingLatticeInvT) -> None:
        """Sets a lattice to its most pessimistic (entry) state."""
        ...
