"""
We store the register that a value will be stored in at runtime on the value's type.
When the value is unallocated, it is considered to be in one of an unbounded set of
registers, meaning each register only ever holds one value.
If it is allocated, then at no point should two values with the same allocated register
type be "live" at the same point in the IR.
The infrastructure in this class helps detect when this occurs, used for verification
and legalization.
"""

from __future__ import annotations

import abc
from dataclasses import KW_ONLY, dataclass, replace

from typing_extensions import Self

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.utils.exceptions import PassFailedException, VerifyException


@dataclass
class LivenessContext(abc.ABC):
    """
    Reverse-liveness walk over register-allocatable IR.
    On entry to `update_liveness`, `alive` holds the values live after the operation
    (its results already removed), on exit it must hold those live before it.
    Use `copy` to fork `alive` for a nested region walk while sharing other state.
    """

    _ = KW_ONLY
    alive: set[SSAValue]

    def copy(self, alive: set[SSAValue]) -> Self:
        """Fork liveness state for a nested region walk; shares all other fields."""
        return replace(self, alive=alive)

    @abc.abstractmethod
    def handle_live_inout(
        self, op: Operation, value: SSAValue, *, duplicate_inout: bool = False
    ) -> SSAValue:
        """
        `op` is about to clobber `value`, which is still live, or `value` is used by
        more than one in/out operand when `duplicate_inout` is True.
        Subclasses must override this to either raise an error or insert a copy to avoid
        the clobber.
        """

    def process_region(self, region: Region) -> None:
        if region.first_block is None:
            return
        if len(region.blocks) > 1:
            raise PassFailedException(
                "Cannot yet verify register liveness for regions with multiple blocks."
            )
        self.process_block(region.first_block)

    def process_block(self, block: Block) -> None:
        for op in reversed(block.ops):
            self.alive.difference_update(op.results)
            self.process_op(op)
        self.alive.difference_update(block.args)

    def process_op(self, op: Operation) -> None:
        if isinstance(op, RegisterAllocatableOperation):
            op.update_liveness(self)
        elif op.regions:
            raise PassFailedException(
                f"Cannot verify register liveness through {op.name}: operations with "
                "regions must implement RegisterAllocatableOperation.update_liveness."
            )
        else:
            self.alive.update(op.operands)


@dataclass
class VerifyLivenessContext(LivenessContext):
    """
    Helper to verify that registers can be allocated in the input, raising a
    `VerifyException` if not.
    """

    def handle_live_inout(
        self, op: Operation, value: SSAValue, *, duplicate_inout: bool = False
    ) -> SSAValue:
        """
        Handles live inout by raising a `VerifyException`, with distinct messages for
        the case of duplicate inout or use after using as inout operand.
        """
        if duplicate_inout:
            name_string = (
                f"Value %{value.name_hint}" if value.name_hint is not None else "Value"
            )
            op.emit_error(
                name_string + " is used by more than one in/out operand",
                VerifyException(),
            )
        op.emit_error(
            f"{value.name_hint} should not be read after in/out usage",
            VerifyException(),
        )
