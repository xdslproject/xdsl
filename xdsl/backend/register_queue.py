from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

from xdsl.backend.register_type import RegisterType

_T = TypeVar("_T", bound=RegisterType)


@dataclass
class RegisterQueue(Generic[_T], ABC):
    """
    LIFO queue of registers available for allocation.
    """

    @abstractmethod
    def push(self, reg: _T) -> None:
        """
        Return a register to be made available for allocation.
        """
        ...

    @abstractmethod
    def pop(self, reg_type: type[_T]) -> _T:
        """
        Get the next available register for allocation.
        """
        ...

    @contextmanager
    def reserve_registers(self, regs: Sequence[_T]):
        for reg in regs:
            self.reserve_register(reg)

        yield

        for reg in regs:
            self.unreserve_register(reg)

    @abstractmethod
    def reserve_register(self, reg: _T) -> None:
        """
        Increase the reservation count for a register.
        If the reservation count is greater than 0, a register cannot be pushed back onto
        the queue.
        It is invalid to reserve a register that is available, and popping it before
        unreserving a register will result in an AssertionError.
        """
        ...

    @abstractmethod
    def unreserve_register(self, reg: _T) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0, make
        the register available for allocation.
        """
        ...

    @abstractmethod
    def exclude_register(self, reg: _T) -> None:
        """
        Removes regiesters from available set, if present.
        """
        ...
