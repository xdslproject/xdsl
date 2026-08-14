from dataclasses import dataclass

from typing_extensions import override

from xdsl.backend.register_stack import RegisterStack
from xdsl.backend.x86.arch import UNKNOWN, X86Arch
from xdsl.dialects.x86 import registers


@dataclass
class X86RegisterStack(RegisterStack):
    """
    Available x86-specific registers.
    """

    DEFAULT_ALLOCATABLE_REGISTERS = (
        *reversed(registers.Reg64Type.allocatable_registers()),
        *reversed(UNKNOWN.allocatable_vector_registers()),
    )

    @classmethod
    @override
    def default_allocatable_registers(cls):
        return cls.DEFAULT_ALLOCATABLE_REGISTERS

    @classmethod
    def get_for_arch(cls, arch: X86Arch, *, allow_infinite: bool = False):
        """
        Build a stack holding the registers this target can actually encode.

        The upper half of each vector bank, xmm16-31 and ymm16-31, needs EVEX,
        so it is only allocatable on AVX-512. Handing it out on a VEX-only
        target produces assembly that faults on that hardware.
        """
        return cls.get(
            (
                *reversed(registers.Reg64Type.allocatable_registers()),
                *reversed(arch.allocatable_vector_registers()),
            ),
            allow_infinite=allow_infinite,
        )
