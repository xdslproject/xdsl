from dataclasses import dataclass

from typing_extensions import override

from xdsl.backend.register_stack import RegisterStack
from xdsl.dialects.x86 import register


@dataclass
class X86RegisterStack(RegisterStack):
    """
    Available x86-specific registers.
    """

    DEFAULT_ALLOCATABLE_REGISTERS = (
        *reversed(register.GeneralRegisterType.allocatable_registers()),
        *reversed(register.AVX2RegisterType.allocatable_registers()),
    )

    @classmethod
    @override
    def default_allocatable_registers(cls):
        return cls.DEFAULT_ALLOCATABLE_REGISTERS
