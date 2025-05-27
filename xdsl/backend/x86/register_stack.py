from dataclasses import dataclass

from xdsl.backend.register_stack import RegisterStack
from xdsl.backend.register_type import RegisterType
from xdsl.dialects.x86 import register


@dataclass
class X86RegisterStack(RegisterStack):
    """
    Available x86-specific registers.
    """

    DEFAULT_RESERVED_REGISTERS = {
        register.RAX,
        register.RDX,
        register.RSP,
    }

    DEFAULT_AVAILABLE_REGISTERS = (*reversed(register.YMM),)

    def push(self, reg: RegisterType) -> None:
        if (
            isinstance(reg, register.GeneralRegisterType)
            and reg not in self.DEFAULT_RESERVED_REGISTERS
        ):
            raise NotImplementedError("x86 general register type not implemented yet.")
        super().push(reg)

    @classmethod
    def default_reserved_registers(cls):
        return cls.DEFAULT_RESERVED_REGISTERS

    @classmethod
    def default_available_registers(cls):
        return cls.DEFAULT_AVAILABLE_REGISTERS
