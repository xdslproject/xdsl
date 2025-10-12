from __future__ import annotations

from enum import Enum
from typing import Self

from xdsl.analysis.dataflow import DataFlowSolver
from xdsl.analysis.sparse_analysis import (
    AbstractLatticeValue,
    Lattice,
    SparseForwardDataFlowAnalysis,
)
from xdsl.folder import Folder
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.traits import ConstantLike, HasFolder


class _ConstantValueState(Enum):
    """The state of a ConstantValue lattice element."""

    UNINITIALIZED = 0
    CONSTANT = 1
    UNKNOWN = 2


class ConstantValue(AbstractLatticeValue):
    """
    A lattice value representing a potential constant. It can be in one of three
    states: uninitialized, a known constant attribute, or unknown (not a constant).
    """

    _state: _ConstantValueState
    _value: Attribute | None

    def __init__(self):
        self._state = _ConstantValueState.UNINITIALIZED
        self._value = None

    @classmethod
    def uninitialized(cls) -> Self:
        """Creates a `ConstantValue` in the uninitialized state."""
        return cls()

    @classmethod
    def constant(cls, attr: Attribute) -> Self:
        """Creates a `ConstantValue` with a known constant attribute."""
        cv = cls()
        cv._state = _ConstantValueState.CONSTANT
        cv._value = attr
        return cv

    @classmethod
    def unknown(cls) -> Self:
        """Creates a `ConstantValue` in the unknown state."""
        cv = cls()
        cv._state = _ConstantValueState.UNKNOWN
        return cv

    @property
    def is_uninitialized(self) -> bool:
        return self._state == _ConstantValueState.UNINITIALIZED

    @property
    def is_constant(self) -> bool:
        return self._state == _ConstantValueState.CONSTANT

    @property
    def is_unknown(self) -> bool:
        return self._state == _ConstantValueState.UNKNOWN

    @property
    def value(self) -> Attribute:
        """Returns the constant attribute. Raises an error if not in a constant state."""
        if not self.is_constant or self._value is None:
            raise RuntimeError("ConstantValue is not a constant")
        return self._value

    def join(self: Self, other: Self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """The join operation for the constant value lattice."""
        if self.is_uninitialized:
            return other
        if other.is_uninitialized:
            return self
        if self.is_unknown or other.is_unknown:
            return type(self).unknown()
        # Both are constants
        if self.value == other.value:
            return self
        return type(self).unknown()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstantValue):
            return NotImplemented
        if self._state != other._state:
            return False
        if self.is_constant:
            return self._value == other._value
        return True

    def __str__(self) -> str:
        if self.is_uninitialized:
            return "<UNINITIALIZED>"
        if self.is_unknown:
            return "<UNKNOWN>"
        return f"const({self.value})"


class SparseConstantPropagation(SparseForwardDataFlowAnalysis[Lattice[ConstantValue]]):
    """
    An analysis that implements sparse constant propagation.
    It determines if an SSA value is a constant, and if so, what its value is.
    """

    folder: Folder

    def __init__(self, solver: DataFlowSolver):
        # The parent constructor is called by the solver's `load` method.
        # We get the context from the solver, which is needed by the Folder.
        super().__init__(solver, Lattice[ConstantValue])
        self.folder = Folder(self.solver.context)

    def set_to_entry_state(self, lattice: Lattice[ConstantValue]) -> None:
        """
        The entry state for any value is 'unknown', as we cannot make any
        assumptions about its value.
        """
        assert isinstance(lattice.anchor, SSAValue)
        self.join(lattice, Lattice(lattice.anchor, ConstantValue.unknown()))

    def visit_operation_impl(
        self,
        op: Operation,
        operands: list[Lattice[ConstantValue]],
        results: list[Lattice[ConstantValue]],
    ) -> None:
        # If the op is a constant, its result is constant.
        if (trait := op.get_trait(ConstantLike)) is not None:
            new_const_val = trait.get_constant_value(op)
            if new_const_val is not None:
                assert isinstance(results[0].anchor, SSAValue)
                self.join(
                    results[0],
                    Lattice(results[0].anchor, ConstantValue.constant(new_const_val)),
                )
            return

        # 1. Collect constant operand values from the lattices.
        const_attrs: list[Attribute] = []
        for operand_lattice in operands:
            val = operand_lattice.value
            if val.is_uninitialized:
                # An input is not ready, bail out and wait for it to be resolved.
                return
            if not val.is_constant:
                # An input is 'unknown', so all results are 'unknown'.
                self.set_all_to_entry_state(results)
                return
            const_attrs.append(val.value)

        # Check if the operation implements HasFolderInterface
        if (trait := op.get_trait(HasFolder)) is None:
            self.set_all_to_entry_state(results)
            return

        # Call fold directly on the operation
        fold_results = trait.fold(op)

        if fold_results is None:
            # Folding failed; results are 'unknown'.
            self.set_all_to_entry_state(results)
            return

        if len(fold_results) != len(results):
            self.set_all_to_entry_state(results)
            return

        # Propagate fold results to the result lattices.
        # Only attribute results are const-propped; SSAValue results are unknown.
        for res_lattice, folded_val in zip(results, fold_results):
            if isinstance(folded_val, Attribute):
                # Attribute result - propagate as constant
                new_const_val = ConstantValue.constant(folded_val)
                assert isinstance(res_lattice.anchor, SSAValue)
                self.join(res_lattice, Lattice(res_lattice.anchor, new_const_val))
            else:
                # SSAValue result - set to unknown/entry state
                self.set_to_entry_state(res_lattice)
