from dataclasses import dataclass

from typing_extensions import override

from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects.riscv import FloatRegisterType, IntRegisterType


@dataclass
class RiscvRegisterStack(RegisterStack):
    """
    Available RISCV-specific registers.
    """

    DEFAULT_ALLOCATABLE_REGISTERS = (
        *reversed(IntRegisterType.allocatable_registers()),
        *reversed(FloatRegisterType.allocatable_registers()),
    )

    @classmethod
    @override
    def default_allocatable_registers(cls):
        return RiscvRegisterStack.DEFAULT_ALLOCATABLE_REGISTERS
