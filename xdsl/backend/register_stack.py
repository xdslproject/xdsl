from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from typing_extensions import TypeVar

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import IntAttr, NoneAttr
from xdsl.utils.exceptions import DiagnosticException

_T = TypeVar("_T", bound=RegisterType)


class OutOfRegisters(DiagnosticException):
    def __str__(self):
        return "Out of registers."


@dataclass
class RegisterStack:
    """
    LIFO stack of registers available for allocation.

    Internal maps are keyed by
    [RegisterType.register_pool_key][xdsl.backend.register_type.RegisterType.register_pool_key]
    so targets can merge pools for aliasing registers.
    """

    allocatable_registers: defaultdict[str, set[int]] = field(
        default_factory=lambda: defaultdict(lambda: set[int]())
    )
    """
    Registers that can be used by the register allocator for a given allocation pool.
    """

    next_infinite_indices: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(lambda: 0)
    )
    """Next index to be used for a given allocation pool."""

    reserved_registers: defaultdict[str, defaultdict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict[int, int](lambda: 0))
    )
    """
    Registers unavailable to be used by the register allocator for a given pool.
    """

    available_registers: defaultdict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list[int])
    )
    """
    Registers from a given pool whose values can be allocated to in the current context.
    """

    allow_infinite: bool = False
    """
    When there are no more registers, use infinite register set.
    """

    @classmethod
    def default_allocatable_registers(cls) -> Iterable[RegisterType]:
        """
        The default registers to be made available when instantiating the stack.
        """
        return ()

    @classmethod
    def get(
        cls,
        allocatable_registers: Iterable[RegisterType] | None = None,
        *,
        allow_infinite: bool = False,
    ):
        if allocatable_registers is None:
            allocatable_registers = cls.default_allocatable_registers()
        res = cls(allow_infinite=allow_infinite)
        for reg in allocatable_registers:
            res.include_register(reg)
        return res

    def push(self, reg: RegisterType) -> None:
        """
        Return a register to be made available for allocation.
        """
        if not isinstance(reg.index, IntAttr):
            raise ValueError("Cannot push an unallocated register")

        index = reg.index.data
        pool_key = reg.register_pool_key()
        if (
            index in self.reserved_registers[pool_key]
            or index not in self.allocatable_registers[pool_key]
        ) and 0 <= index:
            return

        available = self.available_registers[pool_key]
        if index in available:
            available.remove(index)
        available.append(index)

    def pop(self, reg_type: type[_T]) -> _T:
        """
        Get the next available register for allocation.
        """
        pool_key = reg_type.register_pool_key()
        pool = self.available_registers[pool_key]

        if pool:
            reg = reg_type.from_index(pool.pop())
        else:
            if self.allow_infinite:
                reg = reg_type.infinite_register(self.next_infinite_indices[pool_key])
                self.next_infinite_indices[pool_key] += 1
            else:
                raise OutOfRegisters

        reserved_registers = self.reserved_registers[pool_key]

        assert isinstance(reg.index, IntAttr)
        assert reg.index.data not in reserved_registers, (
            f"Cannot pop a reserved register ({reg.register_name.data}), it must have been reserved while available."
        )
        return reg

    def reserve_register(self, reg: RegisterType) -> None:
        """
        Increase the reservation count for a register.
        If the reservation count is greater than 0, a register cannot be pushed back
        onto the stack.
        It is invalid to reserve a register that is available, and popping it before
        unreserving a register will result in an AssertionError.
        """
        assert isinstance(reg.index, IntAttr)
        self.reserved_registers[reg.register_pool_key()][reg.index.data] += 1

    @contextmanager
    def reserve_registers(self, regs: Sequence[RegisterType]):
        for reg in regs:
            self.reserve_register(reg)

        yield

        for reg in regs:
            self.unreserve_register(reg)

    def unreserve_register(self, reg: RegisterType) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0,
        make the register available for allocation.
        """
        assert isinstance(reg.index, IntAttr)
        pool_key = reg.register_pool_key()
        reserved_registers = self.reserved_registers[pool_key]
        if reg.index.data not in reserved_registers:
            raise ValueError(f"Cannot unreserve register {reg.register_name}")
        reserved_registers[reg.index.data] -= 1
        if not reserved_registers[reg.index.data]:
            del reserved_registers[reg.index.data]

    def include_register(self, reg: RegisterType) -> None:
        """
        Makes register available for allocation.
        """
        assert not isinstance(reg.index, NoneAttr)
        pool_key = reg.register_pool_key()
        self.allocatable_registers[pool_key].add(reg.index.data)
        self.push(reg)

    def exclude_register(self, reg: RegisterType) -> None:
        """
        Removes register from available set, if present.
        """
        assert isinstance(reg.index, IntAttr)
        index = reg.index.data
        pool_key = reg.register_pool_key()
        available_registers = self.available_registers[pool_key]
        allocatable_registers = self.allocatable_registers[pool_key]
        if index in available_registers:
            available_registers.remove(index)
        if index in allocatable_registers:
            allocatable_registers.remove(reg.index.data)
