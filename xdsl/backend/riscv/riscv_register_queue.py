from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import overload

from xdsl.backend.register_queue import RegisterQueue
from xdsl.dialects.builtin import IntAttr
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

    reserved_int_registers: defaultdict[int, int] = field(
        default_factory=lambda: defaultdict[int, int](lambda: 0)
    )
    "Integer registers unavailable to be used by the register allocator."

    reserved_float_registers: defaultdict[int, int] = field(
        default_factory=lambda: defaultdict[int, int](lambda: 0)
    )
    "Floating-point registers unavailable to be used by the register allocator."

    available_int_registers: list[int] = field(default_factory=list)
    "Registers that integer values can be allocated to in the current context."

    available_float_registers: list[int] = field(default_factory=list)
    "Registers that floating-point values can be allocated to in the current context."

    @classmethod
    def default(
        cls,
        reserved_registers: Iterable[IntRegisterType | FloatRegisterType] | None = None,
        available_registers: Iterable[IntRegisterType | FloatRegisterType]
        | None = None,
    ):
        if reserved_registers is None:
            reserved_registers = RiscvRegisterQueue.DEFAULT_RESERVED_REGISTERS
        if available_registers is None:
            available_registers = (
                RiscvRegisterQueue.DEFAULT_INT_REGISTERS
                + RiscvRegisterQueue.DEFAULT_FLOAT_REGISTERS
            )
        res = cls()
        for reg in reserved_registers:
            res.reserve_register(reg)
        for reg in available_registers:
            res.push(reg)
        return res

    def push(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Return a register to be made available for allocation.
        """
        if not isinstance(reg.index, IntAttr):
            raise ValueError("Cannot push an unallocated register")

        if isinstance(reg, IntRegisterType):
            if reg.index.data in self.reserved_int_registers:
                return
            self.available_int_registers.append(reg.index.data)
        else:
            if reg.index.data in self.reserved_float_registers:
                return
            self.available_float_registers.append(reg.index.data)

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
            reg = reg_type.from_index(available_registers.pop())
        else:
            if issubclass(reg_type, IntRegisterType):
                reg = reg_type.infinite_register(self._j_idx)
                self._j_idx += 1
            else:
                reg = reg_type.infinite_register(self._fj_idx)
                self._fj_idx += 1

        reserved_registers = (
            self.reserved_int_registers
            if issubclass(reg_type, IntRegisterType)
            else self.reserved_float_registers
        )

        assert isinstance(reg.index, IntAttr)
        assert reg.index.data not in reserved_registers, (
            f"Cannot pop a reserved register ({reg.register_name.data}), it must have been reserved while available."
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
        assert isinstance(reg.index, IntAttr)
        if isinstance(reg, IntRegisterType):
            self.reserved_int_registers[reg.index.data] += 1
        if isinstance(reg, FloatRegisterType):
            self.reserved_float_registers[reg.index.data] += 1

    def unreserve_register(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Decrease the reservation count for a register. If the reservation count is 0, make
        the register available for allocation.
        """
        assert isinstance(reg.index, IntAttr)
        if isinstance(reg, IntRegisterType):
            if reg.index.data not in self.reserved_int_registers:
                raise ValueError(f"Cannot unreserve register {reg.register_name}")
            self.reserved_int_registers[reg.index.data] -= 1
            if not self.reserved_int_registers[reg.index.data]:
                del self.reserved_int_registers[reg.index.data]
        if isinstance(reg, FloatRegisterType):
            if reg.index.data not in self.reserved_float_registers:
                raise ValueError(f"Cannot unreserve register {reg.register_name}")
            self.reserved_float_registers[reg.index.data] -= 1
            if not self.reserved_float_registers[reg.index.data]:
                del self.reserved_float_registers[reg.index.data]

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
        assert isinstance(reg.index, IntAttr)
        if (
            isinstance(reg, IntRegisterType)
            and reg.index.data in self.available_int_registers
        ):
            self.available_int_registers.remove(reg.index.data)
        if (
            isinstance(reg, FloatRegisterType)
            and reg.index.data in self.available_float_registers
        ):
            self.available_float_registers.remove(reg.index.data)
