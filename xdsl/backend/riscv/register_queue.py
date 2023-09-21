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

    reserved_registers: set[IntRegisterType | FloatRegisterType] = field(
        default_factory=lambda: set(RegisterQueue.DEFAULT_RESERVED_REGISTERS)
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
    def pop(self, reg_type: type[IntRegisterType]) -> IntRegisterType:
        ...

    @overload
    def pop(self, reg_type: type[FloatRegisterType]) -> FloatRegisterType:
        ...

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
        return reg

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
