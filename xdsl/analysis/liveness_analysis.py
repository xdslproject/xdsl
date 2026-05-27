"""
Liveness analysis: a sparse backward dataflow analysis that, for every SSA
value, determines whether it is "live".

A value is considered
"live" iff it:

  (1) has memory effects, OR
  (2) is returned by a public function, OR
  (3) is used to compute a value of type (1) or (2), OR
  (4) is returned by a return-like op whose parent isn't a callable nor a
      `RegionBranchOpInterface` (e.g. `linalg.yield`, `gpu.yield`, ...).
      Such ops have op-specific semantics, so we conservatively mark the
      yielded value live.

Note: support for branch operands, call operands, and non-control-flow region
arguments are not yet implemented. They should be added together with the
corresponding op interfaces (`BranchOpInterface`, `CallOpInterface`, `RegionBranchOpInterface`).
"""

from __future__ import annotations

from xdsl.analysis.dataflow import ChangeResult, DataFlowSolver
from xdsl.analysis.sparse_analysis import (
    PropagatingLattice,
    SparseBackwardDataFlowAnalysis,
)
from xdsl.ir import Operation, SSAValue
from xdsl.transforms.dead_code_elimination import would_be_trivially_dead


class Liveness(PropagatingLattice):
    """
    Boolean liveness lattice attached to an SSA value.

    Starts pessimistic (`is_live = False`) and only ever moves towards `True`.
    The lattice ordering is `dead ⊑ live`; `meet` is OR — a value is live if
    either side says so.
    """

    is_live: bool

    def __init__(self, anchor: SSAValue):
        super().__init__(anchor)
        self.is_live = False

    def mark_live(self) -> ChangeResult:
        if self.is_live:
            return ChangeResult.NO_CHANGE
        self.is_live = True
        return ChangeResult.CHANGE

    def meet(self, other: Liveness) -> ChangeResult:
        if other.is_live:
            return self.mark_live()
        return ChangeResult.NO_CHANGE

    def join(self, other: Liveness) -> ChangeResult:
        # Not used for backward analyses, but provided for completeness.
        return self.meet(other)

    def __str__(self) -> str:
        return "live" if self.is_live else "not live"


class LivenessAnalysis(SparseBackwardDataFlowAnalysis[Liveness]):
    """
    Backward sparse dataflow analysis annotating each SSA value with whether it
    is live.
    """

    def __init__(self, solver: DataFlowSolver):
        super().__init__(solver, Liveness)

    def visit_operation_impl(
        self,
        op: Operation,
        operand_lattices: list[Liveness],
        result_lattices: list[Liveness],
    ) -> None:
        # (1.a)/(4) If the op is not trivially dead (has side effects, is a
        # terminator, is a symbol op, ...), every operand is live.
        if not would_be_trivially_dead(op):
            for operand in operand_lattices:
                self.propagate_if_changed(operand, operand.mark_live())

        # (3) If any result is live, every operand is live. The assumption
        # — matching MLIR — is that every operand can contribute to every
        # result, so a single live result suffices to mark all operands live.
        for result in result_lattices:
            if result.is_live:
                for operand in operand_lattices:
                    self.meet(operand, result)
                break

    def set_to_exit_state(self, lattice: Liveness) -> None:
        """
        Called for lattices that reach a backward-propagation boundary
        (e.g. operands of a public-function return). Marks them live (2).
        """
        if lattice.is_live:
            return
        lattice.is_live = True
        self.propagate_if_changed(lattice, ChangeResult.CHANGE)
