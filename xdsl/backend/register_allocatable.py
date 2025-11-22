import abc
from collections.abc import Iterator, Sequence
from typing import NamedTuple

from xdsl.backend.register_allocator import BlockAllocator
from xdsl.backend.register_type import RegisterType
from xdsl.ir import Operation, Region, SSAValue
from xdsl.irdl import traits_def
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import VerifyException


class RegisterAllocatableOperation(Operation, abc.ABC):
    """
    An abstract base class for operations that can be processed during register
    allocation.
    """

    def iter_used_registers(self) -> Iterator[RegisterType]:
        """
        The registers whose contents may be overwritten when executing this operation.
        By default returns the types of operands and results that are allocated
        registers.
        """
        return (
            val.type
            for vals in (self.operands, self.results)
            for val in vals
            if isinstance(val.type, RegisterType) and val.type.is_allocated
        )

    @abc.abstractmethod
    def allocate_registers(self, allocator: BlockAllocator) -> None:
        """
        Allocate registers for this operation.
        """

    @staticmethod
    def iter_all_used_registers(
        region: Region,
    ) -> Iterator[RegisterType]:
        """
        All used registers of all operations within a region.
        """
        return (
            reg
            for op in region.walk()
            if isinstance(op, RegisterAllocatableOperation)
            for reg in op.iter_used_registers()
        )


class RegisterConstraints(NamedTuple):
    """
    Values used by an instruction.
    A collection of operations in `inouts` represents the constraint that they must be
    allocated to the same register.
    """

    ins: Sequence[SSAValue]
    outs: Sequence[SSAValue]
    inouts: Sequence[Sequence[SSAValue]]


class HasRegisterConstraintsTrait(OpTrait):
    """
    Trait that verifies that the operation implements HasRegisterConstraints, and that
    its inout operands are used only once.
    Using an inout operand more than once breaks SSA, as the register will hold an
    unexpected value after being mutated by this operation.
    """

    def verify(self, op: Operation) -> None:
        if not isinstance(op, HasRegisterConstraints):
            raise VerifyException(
                f"Operation {op.name} is not a subclass of {HasRegisterConstraints.__name__}."
            )


class HasRegisterConstraints(RegisterAllocatableOperation, abc.ABC):
    """
    Abstract superclass for operations corresponding to assembly, with registers used
    as in, out, or inout registers.
    The use of a register value as inout must be its last use (externally verified,
    e.g. for x86 see pass verify-register-allocation).
    """

    traits = traits_def(HasRegisterConstraintsTrait())

    @abc.abstractmethod
    def get_register_constraints(self) -> RegisterConstraints:
        """
        The values with register types used by this operation, for use in register
        allocation.
        """
        raise NotImplementedError()

    def allocate_registers(self, allocator: BlockAllocator) -> None:
        ins, outs, inouts = self.get_register_constraints()

        # Allocate registers to inout operand groups since they are defined further up
        # in the use-def SSA chain
        for operand_group in inouts:
            allocator.allocate_values_same_reg(operand_group)

        for result in outs:
            # Allocate registers to result if not already allocated
            if (new_result := allocator.allocate_value(result)) is not None:
                result = new_result
            allocator.free_value(result)

        # Allocate registers to operands since they are defined further up
        # in the use-def SSA chain
        for operand in ins:
            allocator.allocate_value(operand)
