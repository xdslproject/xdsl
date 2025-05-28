from dataclasses import dataclass

from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects.riscv import Registers


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
        *reversed(Registers.A),
        *reversed(Registers.T),
        *reversed(Registers.FA),
        *reversed(Registers.FT),
    )

    @classmethod
    def default_reserved_registers(cls):
        return RiscvRegisterStack.DEFAULT_RESERVED_REGISTERS

    @classmethod
    def default_available_registers(cls):
        return RiscvRegisterStack.DEFAULT_AVAILABLE_REGISTERS
