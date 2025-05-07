from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeVar

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntAttr

_T = TypeVar("_T", bound=RegisterType)


@dataclass
class RegisterQueue(ABC):
    """
    LIFO queue of registers available for allocation.
    """

    @abstractmethod
    def push(self, reg: RegisterType) -> None:
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
    def reserve_registers(self, regs: Sequence[RegisterType]):
        for reg in regs:
            self.reserve_register(reg)

        yield

        for reg in regs:
            self.unreserve_register(reg)

    @abstractmethod
    def reserve_register(self, reg: RegisterType) -> None:
        """
        Increase the reservation count for a register.
        If the reservation count is greater than 0, a register cannot be pushed back onto
        the queue.
        It is invalid to reserve a register that is available, and popping it before
        unreserving a register will result in an AssertionError.
        """
        ...

    @abstractmethod
    def unreserve_register(self, reg: RegisterType) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0, make
        the register available for allocation.
        """
        ...

    @abstractmethod
    def exclude_register(self, reg: RegisterType) -> None:
        """
        Removes regiesters from available set, if present.
        """
        ...


@dataclass
class LIFORegisterQueue(RegisterQueue):
    """
    LIFO queue of registers available for allocation.
    """

    next_infinite_indices: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(lambda: 0)
    )
    """Next index for a given register set."""

    reserved_registers: defaultdict[str, defaultdict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict[int, int](lambda: 0))
    )
    """
    Registers unavailable to be used by the register allocator for a given register set.
    """

    available_registers: defaultdict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list[int])
    )
    """
    Registers from a given register set that values can be allocated to in the current
    context.
    """

    @classmethod
    def default_reserved_registers(cls) -> Iterable[RegisterType]:
        """
        The default registers to be made unavailable when instantiating the queue.
        """
        return ()

    @classmethod
    def default_available_registers(cls) -> Iterable[RegisterType]:
        """
        The default registers to be made available when instantiating the queue.
        """
        return ()

    @classmethod
    def default(
        cls,
        reserved_registers: Iterable[RegisterType] | None = None,
        available_registers: Iterable[RegisterType] | None = None,
    ):
        if reserved_registers is None:
            reserved_registers = cls.default_reserved_registers()
        if available_registers is None:
            available_registers = cls.default_available_registers()
        res = cls()
        for reg in reserved_registers:
            res.reserve_register(reg)
        for reg in available_registers:
            res.push(reg)
        return res

    def push(self, reg: RegisterType) -> None:
        """
        Return a register to be made available for allocation.
        """
        if not isinstance(reg.index, IntAttr):
            raise ValueError("Cannot push an unallocated register")

        register_set = reg.name
        if reg.index.data in self.reserved_registers[register_set]:
            return

        self.available_registers[register_set].append(reg.index.data)

    def pop(self, reg_type: type[_T]) -> _T:
        """
        Get the next available register for allocation.
        """
        register_set = reg_type.name
        available_registers = self.available_registers[register_set]

        if available_registers:
            reg = reg_type.from_index(available_registers.pop())
        else:
            reg = reg_type.infinite_register(self.next_infinite_indices[register_set])
            self.next_infinite_indices[register_set] += 1

        reserved_registers = self.reserved_registers[reg_type.name]

        assert isinstance(reg.index, IntAttr)
        assert reg.index.data not in reserved_registers, (
            f"Cannot pop a reserved register ({reg.register_name.data}), it must have been reserved while available."
        )
        return reg

    def reserve_register(self, reg: RegisterType) -> None:
        """
        Increase the reservation count for a register.
        If the reservation count is greater than 0, a register cannot be pushed back onto
        the queue.
        It is invalid to reserve a register that is available, and popping it before
        unreserving a register will result in an AssertionError.
        """
        assert isinstance(reg.index, IntAttr)
        self.reserved_registers[reg.name][reg.index.data] += 1

    def unreserve_register(self, reg: RegisterType) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0, make
        the register available for allocation.
        """
        assert isinstance(reg.index, IntAttr)
        reserved_registers = self.reserved_registers[reg.name]
        if reg.index.data not in reserved_registers:
            raise ValueError(f"Cannot unreserve register {reg.register_name}")
        reserved_registers[reg.index.data] -= 1
        if not reserved_registers[reg.index.data]:
            del reserved_registers[reg.index.data]

    def limit_registers(self, limit: int) -> None:
        """
        Limits the number of currently available registers to the provided limit.
        """
        if limit < 0:
            raise ValueError(f"Invalid negative limit value {limit}")

        keys = tuple(self.available_registers)
        if limit:
            for key in keys:
                self.available_registers[key] = self.available_registers[key][-limit:]
        else:
            for key in keys:
                del self.available_registers[key]

    def exclude_register(self, reg: RegisterType) -> None:
        """
        Removes register from available set, if present.
        """
        assert isinstance(reg.index, IntAttr)
        available_registers = self.available_registers[reg.name]
        if reg.index.data in available_registers:
            available_registers.remove(reg.index.data)
