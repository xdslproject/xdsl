import abc
from collections.abc import Iterator, Sequence
from typing import NamedTuple

from xdsl.backend.register_type import RegisterType
from xdsl.ir import Operation, Region, SSAValue


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


class HasRegisterConstraints(RegisterAllocatableOperation, abc.ABC):
    @abc.abstractmethod
    def get_register_constraints(self) -> RegisterConstraints:
        """
        The values with register types used by this operation, for use in register
        allocation.
        """
        raise NotImplementedError()
