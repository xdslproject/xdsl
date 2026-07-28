from __future__ import annotations

import abc
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import deprecated

from xdsl.backend.register_allocator import BlockAllocator
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect, RegisterType
from xdsl.ir import Operation, OpResult, Region, SSAValue, SSAValueCovT
from xdsl.irdl import traits_def
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.backend.liveness import LivenessContext


class RegisterAllocatableOperation(Operation, abc.ABC):
    """
    An abstract base class for operations that can be processed during register
    allocation.
    """

    @deprecated("Use register effects instead")
    def iter_used_registers(self) -> Iterator[RegisterType]:
        """
        The registers whose contents may be overwritten when executing this operation.
        By default returns the types of operands and results that are allocated
        registers.
        """
        yield from ()

    def iter_excluded_registers(self) -> Iterator[RegisterType]:
        """
        The registers that should not be used when this operation is present.
        """
        yield from ()

    @abc.abstractmethod
    def allocate_registers(self, allocator: BlockAllocator) -> None:
        """
        Allocate registers for this operation.
        """

    @abc.abstractmethod
    def update_liveness(self, ctx: LivenessContext) -> None:
        """
        Update `ctx.alive` from live-after to live-before this operation.
        """

    @staticmethod
    def all_used_registers(
        region: Region,
    ) -> AbstractSet[RegisterType]:
        """
        All used registers of all operations within a region.
        """
        return {
            reg
            for op in region.walk()
            if isinstance(op, RegisterAllocatableOperation)
            for reg in RegisterAllocatedMemoryEffect.iter_used_registers(op)
        }

    @staticmethod
    def all_excluded_registers(
        region: Region,
    ) -> AbstractSet[RegisterType]:
        """
        All excluded registers as declared by all operations within a region.
        """
        return {
            reg
            for op in region.walk()
            if isinstance(op, RegisterAllocatableOperation)
            for reg in op.iter_excluded_registers()
        }


class RegisterConstraints(NamedTuple):
    """
    Values used by an instruction.
    A collection of operations in `inouts` represents the constraint that they must be
    allocated to the same register.
    """

    ins: Sequence[SSAValue]
    outs: Sequence[OpResult]
    inouts: Sequence[tuple[SSAValue, OpResult]]


def _verify_declared_once(
    op_name: str,
    kind: str,
    values: Sequence[SSAValueCovT],
    declared: Mapping[SSAValueCovT, int],
    roles: str,
) -> None:
    """
    Verify that each of `values` is declared as many times as it occurs.
    Counting occurrences rather than testing membership lets a value taking several
    roles be declared once per role, as when an operation reads a value and also
    clobbers it.
    """
    occurrences = Counter(values)
    if declared == occurrences:
        return
    for index, value in enumerate(values):
        if declared[value] != occurrences[value]:
            raise VerifyException(
                f"Operation {op_name} {kind} at index {index} is declared "
                f"{declared[value]} times as {roles}, expected {occurrences[value]}."
            )
    raise VerifyException(
        f"Operation {op_name} declares values as {roles} that it does not use as "
        f"{kind}s."
    )


class HasRegisterConstraintsTrait(OpTrait):
    """
    Trait that verifies that the operation implements HasRegisterConstraints, and that
    its constraints account for each operand and result as many times as it occurs.
    An operand is declared by `ins` or `inouts`, a result by `outs` or `inouts`. A value
    occupying several operands is declared once per operand, so an operation may read a
    value as an `in` register and also clobber it as an `inout` register.
    """

    def verify(self, op: Operation) -> None:
        if not isinstance(op, HasRegisterConstraints):
            raise VerifyException(
                f"Operation {op.name} is not a subclass of {HasRegisterConstraints.__name__}."
            )
        ins, outs, inouts = op.get_register_constraints()

        declared_operands = Counter(ins)
        declared_results = Counter(outs)
        # A value cannot be both an operand and a result of the same operation, so
        # membership tells which side of the constraint each inout value belongs to.
        for operand, result in inouts:
            declared_operands[operand] += 1
            declared_results[result] += 1
        _verify_declared_once(
            op.name, "operand", op.operands, declared_operands, "`in` or `inout`"
        )
        _verify_declared_once(
            op.name, "result", op.results, declared_results, "`out` or `inout`"
        )


class HasRegisterConstraints(RegisterAllocatableOperation, abc.ABC):
    """
    Abstract superclass for operations corresponding to assembly, with registers used
    as in, out, or inout registers.
    The use of a register value as inout must be its last use (externally verified,
    e.g. see pass x86-regalloc-verify-liveness).
    """

    traits = traits_def(HasRegisterConstraintsTrait())

    @abc.abstractmethod
    def get_register_constraints(self) -> RegisterConstraints:
        """
        The values with register types used by this operation, for use in register
        allocation.
        """
        raise NotImplementedError()

    def update_liveness(self, ctx: LivenessContext) -> None:
        ins, _, inouts = self.get_register_constraints()
        clobbered: set[SSAValue] = set()
        for operand, _ in inouts:
            # Each inout slot needs its own register, so a value already claimed by an
            # earlier slot must be handled even when nothing reads it afterwards.
            use_after_inout = operand in ctx.alive
            duplicate_inout = operand in clobbered
            if use_after_inout or duplicate_inout:
                new_operand = ctx.handle_live_inout(
                    self, operand, duplicate_inout=duplicate_inout
                )
                # Replacing a position clears the value from it, so the first
                # position still holding it is the one for this slot.
                self.operands[self.operands.index(operand)] = new_operand
            clobbered.add(operand)
        # The constraints were read before any replacement, so a replaced operand is
        # still counted here, as it is read by the copy inserted above this operation.
        ctx.alive.update(ins, clobbered)

    def allocate_registers(self, allocator: BlockAllocator) -> None:
        ins, outs, inouts = self.get_register_constraints()

        # Allocate registers to inout operand groups since they are defined further up
        # in the use-def SSA chain
        for operand_group in inouts:
            allocator.allocate_values_same_reg(operand_group)

        new_outs: list[SSAValue] = []
        for result in outs:
            # Allocate registers to result if not already allocated
            if (new_result := allocator.allocate_value(result)) is not None:
                result = new_result
            new_outs.append(result)

        # reverse new_outs to have more optimal allocation in trivial pmov case
        # if all registers are unallocated, this is optimal allocation for pmov
        for result in reversed(new_outs):
            allocator.free_value(result)

        # Allocate registers to operands since they are defined further up
        # in the use-def SSA chain
        for operand in ins:
            allocator.allocate_value(operand)
