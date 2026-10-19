from dataclasses import dataclass

from typing_extensions import override

from xdsl.backend.register_stack import RegisterStack
from xdsl.backend.x86.arch import UNKNOWN, X86Arch


@dataclass
class X86RegisterStack(RegisterStack):
    """
    Available x86-specific registers.
    """

    DEFAULT_ALLOCATABLE_REGISTERS = tuple(
        reversed(UNKNOWN.default_allocatable_registers())
    )

    @classmethod
    @override
    def default_allocatable_registers(cls):
        return cls.DEFAULT_ALLOCATABLE_REGISTERS

    @classmethod
    def get_for_arch(cls, arch: X86Arch, *, allow_infinite: bool = False):
        """
        Build a stack holding the registers this target can actually encode.

        Handing out the EVEX-only half of a vector bank on a VEX-only target
        produces assembly that target cannot run.
        """
        return cls.get(
            reversed(arch.default_allocatable_registers()),
            allow_infinite=allow_infinite,
        )
