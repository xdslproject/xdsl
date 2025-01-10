from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import overload

from xdsl.dialects.riscv import FloatRegisterType, IntRegisterType, Registers


@dataclass
class RegisterQueue:
    """
    LIFO queue of registers available for allocation.
    """

    DEFAULT_RESERVED_REGISTERS = {
        Registers.ZERO,
        Registers.SP,
        Registers.GP,
        Registers.TP,
        Registers.FP,
        Registers.S0,  # Same register as FP
    }

    DEFAULT_INT_REGISTERS = Registers.A[::-1] + Registers.T[::-1]
    DEFAULT_FLOAT_REGISTERS = Registers.FA[::-1] + Registers.FT[::-1]

    _idx: int = 0
    """Next `j` register index."""

    reserved_registers: defaultdict[IntRegisterType | FloatRegisterType, int] = field(
        default_factory=lambda: defaultdict[IntRegisterType | FloatRegisterType, int](
            lambda: 0
        )
        | {r: 1 for r in RegisterQueue.DEFAULT_RESERVED_REGISTERS}
    )
    "Registers unavailable to be used by the register allocator."

    available_int_registers: list[IntRegisterType] = field(
        default_factory=lambda: list(RegisterQueue.DEFAULT_INT_REGISTERS)
    )
    "Registers that integer values can be allocated to in the current context."

    available_float_registers: list[FloatRegisterType] = field(
        default_factory=lambda: list(RegisterQueue.DEFAULT_FLOAT_REGISTERS)
    )
    "Registers that floating-point values can be allocated to in the current context."

    def push(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Return a register to be made available for allocation.
        """
        if reg in self.reserved_registers:
            return
        if not reg.is_allocated:
            raise ValueError("Cannot push an unallocated register")
        if isinstance(reg, IntRegisterType):
            self.available_int_registers.append(reg)
        else:
            self.available_float_registers.append(reg)

    @overload
    def pop(self, reg_type: type[IntRegisterType]) -> IntRegisterType: ...

    @overload
    def pop(self, reg_type: type[FloatRegisterType]) -> FloatRegisterType: ...

    def pop(
        self, reg_type: type[IntRegisterType] | type[FloatRegisterType]
    ) -> IntRegisterType | FloatRegisterType:
        """
        Get the next available register for allocation.
        """
        if issubclass(reg_type, IntRegisterType):
            available_registers = self.available_int_registers
        else:
            available_registers = self.available_float_registers

        if available_registers:
            reg = available_registers.pop()
        else:
            reg = reg_type(f"j{self._idx}")
            self._idx += 1
        assert reg not in self.reserved_registers, (
            f"Cannot pop a reserved register ({reg.register_name}), it must have been reserved while available."
        )
        return reg

    @contextmanager
    def reserve_registers(self, regs: Sequence[IntRegisterType | FloatRegisterType]):
        for reg in regs:
            self.reserve_register(reg)

        yield

        for reg in regs:
            self.unreserve_register(reg)

    def reserve_register(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Increase the reservation count for a register.
        If the reservation count is greater than 0, a register cannot be pushed back onto
        the queue.
        It is invalid to reserve a register that is available, and popping it before
        unreserving a register will result in an AssertionError.
        """
        self.reserved_registers[reg] += 1

    def unreserve_register(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0, make
        the register available for allocation.
        """
        if reg not in self.reserved_registers:
            raise ValueError("Cannot unreserve an unreserved register")
        self.reserved_registers[reg] -= 1
        if not self.reserved_registers[reg]:
            del self.reserved_registers[reg]

    def limit_registers(self, limit: int) -> None:
        """
        Limits the number of currently available registers to the provided limit.
        """
        if limit < 0:
            raise ValueError(f"Invalid negative limit value {limit}")
        if limit:
            self.available_int_registers = self.available_int_registers[-limit:]
            self.available_float_registers = self.available_float_registers[-limit:]
        else:
            self.available_int_registers = []
            self.available_float_registers = []
