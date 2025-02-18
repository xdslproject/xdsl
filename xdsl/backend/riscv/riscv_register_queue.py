from collections import defaultdict
from dataclasses import dataclass, field
from typing import overload

from xdsl.backend.register_queue import RegisterQueue
from xdsl.dialects.riscv import FloatRegisterType, IntRegisterType, Registers


@dataclass
class RiscvRegisterQueue(RegisterQueue[IntRegisterType | FloatRegisterType]):
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

    _j_idx: int = 0
    """Next `j` register index."""

    _fj_idx: int = 0
    """Next `fj` register index."""

    reserved_int_registers: defaultdict[IntRegisterType, int] = field(
        default_factory=lambda: defaultdict[IntRegisterType, int](lambda: 0)
        | {r: 1 for r in RiscvRegisterQueue.DEFAULT_RESERVED_REGISTERS}
    )
    "Integer registers unavailable to be used by the register allocator."

    reserved_float_registers: defaultdict[FloatRegisterType, int] = field(
        default_factory=lambda: defaultdict[FloatRegisterType, int](lambda: 0)
    )
    "Floating-point registers unavailable to be used by the register allocator."

    available_int_registers: list[IntRegisterType] = field(
        default_factory=lambda: list(RiscvRegisterQueue.DEFAULT_INT_REGISTERS)
    )
    "Registers that integer values can be allocated to in the current context."

    available_float_registers: list[FloatRegisterType] = field(
        default_factory=lambda: list(RiscvRegisterQueue.DEFAULT_FLOAT_REGISTERS)
    )
    "Registers that floating-point values can be allocated to in the current context."

    def push(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Return a register to be made available for allocation.
        """
        if reg in self.reserved_int_registers or reg in self.reserved_float_registers:
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
            if issubclass(reg_type, IntRegisterType):
                reg = reg_type(f"j{self._j_idx}")
                self._j_idx += 1
            else:
                reg = reg_type(f"fj{self._fj_idx}")
                self._fj_idx += 1

        reserved_registers = (
            self.reserved_int_registers
            if issubclass(reg_type, IntRegisterType)
            else self.reserved_float_registers
        )

        assert reg not in reserved_registers, (
            f"Cannot pop a reserved register ({reg.register_name}), it must have been reserved while available."
        )
        return reg

    def reserve_register(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Increase the reservation count for a register.
        If the reservation count is greater than 0, a register cannot be pushed back onto
        the queue.
        It is invalid to reserve a register that is available, and popping it before
        unreserving a register will result in an AssertionError.
        """
        if isinstance(reg, IntRegisterType):
            self.reserved_int_registers[reg] += 1
        if isinstance(reg, FloatRegisterType):
            self.reserved_float_registers[reg] += 1

    def unreserve_register(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0, make
        the register available for allocation.
        """
        if isinstance(reg, IntRegisterType):
            if reg not in self.reserved_int_registers:
                raise ValueError(f"Cannot unreserve register {reg.spelling}")
            self.reserved_int_registers[reg] -= 1
            if not self.reserved_int_registers[reg]:
                del self.reserved_int_registers[reg]
        if isinstance(reg, FloatRegisterType):
            if reg not in self.reserved_float_registers:
                raise ValueError(f"Cannot unreserve register {reg.spelling}")
            self.reserved_float_registers[reg] -= 1
            if not self.reserved_float_registers[reg]:
                del self.reserved_float_registers[reg]

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

    def exclude_register(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Removes register from available set, if present.
        """
        if isinstance(reg, IntRegisterType) and reg in self.available_int_registers:
            self.available_int_registers.remove(reg)
        if isinstance(reg, FloatRegisterType) and reg in self.available_float_registers:
            self.available_float_registers.remove(reg)
