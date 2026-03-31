from dataclasses import dataclass

from typing_extensions import override

from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects.x86 import registers


@dataclass
class X86RegisterStack(RegisterStack):
    """
    Available x86-specific registers.
    """

    DEFAULT_ALLOCATABLE_REGISTERS = (
        *reversed(registers.GeneralRegisterType.allocatable_registers()),
        *reversed(registers.AVX2RegisterType.allocatable_registers()),
        *reversed(registers.AVX512RegisterType.allocatable_registers()),
    )

    @classmethod
    @override
    def default_allocatable_registers(cls):
        return cls.DEFAULT_ALLOCATABLE_REGISTERS
