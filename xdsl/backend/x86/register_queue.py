from dataclasses import dataclass

from xdsl.backend.register_queue import LIFORegisterQueue
from xdsl.dialects.x86 import register


@dataclass
class X86RegisterQueue(LIFORegisterQueue):
    """
    LIFO queue of x86-specific registers.
    """

    DEFAULT_RESERVED_REGISTERS = {
        register.RAX,
        register.RDX,
        register.RSP,
    }

    DEFAULT_AVAILABLE_REGISTERS = (*reversed(register.YMM),)

    @classmethod
    def default_reserved_registers(cls):
        return cls.DEFAULT_RESERVED_REGISTERS

    @classmethod
    def default_available_registers(cls):
        return cls.DEFAULT_AVAILABLE_REGISTERS
