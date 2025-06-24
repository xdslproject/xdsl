from dataclasses import dataclass

from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects.riscv import FloatRegisterType, IntRegisterType, Registers


@dataclass
class RiscvRegisterStack(RegisterStack):
    """
    Available RISCV-specific registers.
    """

    DEFAULT_RESERVED_REGISTERS = {
        Registers.ZERO,
        Registers.SP,
        Registers.GP,
        Registers.TP,
        Registers.FP,
        Registers.S0,  # Same register as FP
    }

    DEFAULT_AVAILABLE_REGISTERS = (
        *reversed(IntRegisterType.allocatable_registers()),
        *reversed(FloatRegisterType.allocatable_registers()),
    )

    @classmethod
    def default_reserved_registers(cls):
        return RiscvRegisterStack.DEFAULT_RESERVED_REGISTERS

    @classmethod
    def default_available_registers(cls):
        return RiscvRegisterStack.DEFAULT_AVAILABLE_REGISTERS
