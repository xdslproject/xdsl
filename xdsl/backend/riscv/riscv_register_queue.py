from collections.abc import Iterable
from dataclasses import dataclass

from xdsl.backend.register_queue import LIFORegisterQueue
from xdsl.dialects.riscv import Registers, RISCVRegisterType


@dataclass
class RiscvRegisterQueue(LIFORegisterQueue[RISCVRegisterType]):
    """
    LIFO queue of RISCV-specific registers.
    """

    DEFAULT_RESERVED_REGISTERS = {
        Registers.ZERO,
        Registers.SP,
        Registers.GP,
        Registers.TP,
        Registers.FP,
        Registers.S0,  # Same register as FP
    }

    DEFAULT_AVAILABLE_REGISTERS: tuple[RISCVRegisterType, ...] = (
        *reversed(Registers.A),
        *reversed(Registers.T),
        *reversed(Registers.FA),
        *reversed(Registers.FT),
    )

    @classmethod
    def default_reserved_registers(cls) -> Iterable[RISCVRegisterType]:
        return RiscvRegisterQueue.DEFAULT_RESERVED_REGISTERS

    @classmethod
    def default_available_registers(cls) -> Iterable[RISCVRegisterType]:
        return RiscvRegisterQueue.DEFAULT_AVAILABLE_REGISTERS
