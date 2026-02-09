from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Self, TypeVar

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import NoneAttr
from xdsl.irdl import irdl_attr_definition


class RISCVRegisterType(RegisterType, ABC):
    """
    A RISC-V register type.
    """

    @classmethod
    @abstractmethod
    def a_register(cls, index: int) -> Self:
        raise NotImplementedError()


_RV32I_ABI_INDEX_BY_NAME = {
    "zero": 0,
    "ra": 1,
    "sp": 2,
    "gp": 3,
    "tp": 4,
    "t0": 5,
    "t1": 6,
    "t2": 7,
    "fp": 8,
    "s0": 8,
    "s1": 9,
    "a0": 10,
    "a1": 11,
    "a2": 12,
    "a3": 13,
    "a4": 14,
    "a5": 15,
    "a6": 16,
    "a7": 17,
    "s2": 18,
    "s3": 19,
    "s4": 20,
    "s5": 21,
    "s6": 22,
    "s7": 23,
    "s8": 24,
    "s9": 25,
    "s10": 26,
    "s11": 27,
    "t3": 28,
    "t4": 29,
    "t5": 30,
    "t6": 31,
}
_RV32I_X_INDEX_BY_NAME = {f"x{i}": i for i in range(32)}
RV32I_INDEX_BY_NAME = _RV32I_X_INDEX_BY_NAME | _RV32I_ABI_INDEX_BY_NAME


@irdl_attr_definition
class IntRegisterType(RISCVRegisterType):
    """
    A RISC-V register type.
    """

    name = "riscv.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return RV32I_INDEX_BY_NAME

    @classmethod
    def a_register(cls, index: int) -> IntRegisterType:
        return Registers.A[index]

    @classmethod
    def infinite_register_prefix(cls):
        return "j_"

    # This class variable is created and exclusively accessed in `abi_name_by_index`.
    # _ALLOCATABLE_REGISTERS: ClassVar[tuple[IntRegisterType, ...]]

    @classmethod
    def allocatable_registers(cls):
        if not hasattr(cls, "_ALLOCATABLE_REGISTERS"):
            cls._ALLOCATABLE_REGISTERS = (*Registers.T, *Registers.A)
        return cls._ALLOCATABLE_REGISTERS


_RV32F_ABI_INDEX_BY_NAME = {
    "ft0": 0,
    "ft1": 1,
    "ft2": 2,
    "ft3": 3,
    "ft4": 4,
    "ft5": 5,
    "ft6": 6,
    "ft7": 7,
    "fs0": 8,
    "fs1": 9,
    "fa0": 10,
    "fa1": 11,
    "fa2": 12,
    "fa3": 13,
    "fa4": 14,
    "fa5": 15,
    "fa6": 16,
    "fa7": 17,
    "fs2": 18,
    "fs3": 19,
    "fs4": 20,
    "fs5": 21,
    "fs6": 22,
    "fs7": 23,
    "fs8": 24,
    "fs9": 25,
    "fs10": 26,
    "fs11": 27,
    "ft8": 28,
    "ft9": 29,
    "ft10": 30,
    "ft11": 31,
}
_RV32F_F_INDEX_BY_NAME = {f"f{i}": i for i in range(32)}
RV32F_INDEX_BY_NAME = _RV32F_F_INDEX_BY_NAME | _RV32F_ABI_INDEX_BY_NAME


@irdl_attr_definition
class FloatRegisterType(RISCVRegisterType):
    """
    A RISC-V register type.
    """

    name = "riscv.freg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return RV32F_INDEX_BY_NAME

    @classmethod
    def a_register(cls, index: int) -> FloatRegisterType:
        return Registers.FA[index]

    @classmethod
    def infinite_register_prefix(cls):
        return "fj_"

    # This class variable is created and exclusively accessed in `abi_name_by_index`.
    # _ALLOCATABLE_REGISTERS: ClassVar[tuple[FloatRegisterType, ...]]

    @classmethod
    def allocatable_registers(cls):
        if not hasattr(cls, "_ALLOCATABLE_REGISTERS"):
            cls._ALLOCATABLE_REGISTERS = (*Registers.FT, *Registers.FA)
        return cls._ALLOCATABLE_REGISTERS


RDInvT = TypeVar("RDInvT", bound=RISCVRegisterType)
RSInvT = TypeVar("RSInvT", bound=RISCVRegisterType)
RS1InvT = TypeVar("RS1InvT", bound=RISCVRegisterType)
RS2InvT = TypeVar("RS2InvT", bound=RISCVRegisterType)


class Registers(ABC):
    """Namespace for named register constants."""

    UNALLOCATED_INT = IntRegisterType.unallocated()
    ZERO = IntRegisterType.from_name("zero")
    RA = IntRegisterType.from_name("ra")
    SP = IntRegisterType.from_name("sp")
    GP = IntRegisterType.from_name("gp")
    TP = IntRegisterType.from_name("tp")
    T0 = IntRegisterType.from_name("t0")
    T1 = IntRegisterType.from_name("t1")
    T2 = IntRegisterType.from_name("t2")
    FP = IntRegisterType.from_name("fp")
    S0 = IntRegisterType.from_name("s0")
    S1 = IntRegisterType.from_name("s1")
    A0 = IntRegisterType.from_name("a0")
    A1 = IntRegisterType.from_name("a1")
    A2 = IntRegisterType.from_name("a2")
    A3 = IntRegisterType.from_name("a3")
    A4 = IntRegisterType.from_name("a4")
    A5 = IntRegisterType.from_name("a5")
    A6 = IntRegisterType.from_name("a6")
    A7 = IntRegisterType.from_name("a7")
    S2 = IntRegisterType.from_name("s2")
    S3 = IntRegisterType.from_name("s3")
    S4 = IntRegisterType.from_name("s4")
    S5 = IntRegisterType.from_name("s5")
    S6 = IntRegisterType.from_name("s6")
    S7 = IntRegisterType.from_name("s7")
    S8 = IntRegisterType.from_name("s8")
    S9 = IntRegisterType.from_name("s9")
    S10 = IntRegisterType.from_name("s10")
    S11 = IntRegisterType.from_name("s11")
    T3 = IntRegisterType.from_name("t3")
    T4 = IntRegisterType.from_name("t4")
    T5 = IntRegisterType.from_name("t5")
    T6 = IntRegisterType.from_name("t6")

    UNALLOCATED_FLOAT = FloatRegisterType.unallocated()
    FT0 = FloatRegisterType.from_name("ft0")
    FT1 = FloatRegisterType.from_name("ft1")
    FT2 = FloatRegisterType.from_name("ft2")
    FT3 = FloatRegisterType.from_name("ft3")
    FT4 = FloatRegisterType.from_name("ft4")
    FT5 = FloatRegisterType.from_name("ft5")
    FT6 = FloatRegisterType.from_name("ft6")
    FT7 = FloatRegisterType.from_name("ft7")
    FS0 = FloatRegisterType.from_name("fs0")
    FS1 = FloatRegisterType.from_name("fs1")
    FA0 = FloatRegisterType.from_name("fa0")
    FA1 = FloatRegisterType.from_name("fa1")
    FA2 = FloatRegisterType.from_name("fa2")
    FA3 = FloatRegisterType.from_name("fa3")
    FA4 = FloatRegisterType.from_name("fa4")
    FA5 = FloatRegisterType.from_name("fa5")
    FA6 = FloatRegisterType.from_name("fa6")
    FA7 = FloatRegisterType.from_name("fa7")
    FS2 = FloatRegisterType.from_name("fs2")
    FS3 = FloatRegisterType.from_name("fs3")
    FS4 = FloatRegisterType.from_name("fs4")
    FS5 = FloatRegisterType.from_name("fs5")
    FS6 = FloatRegisterType.from_name("fs6")
    FS7 = FloatRegisterType.from_name("fs7")
    FS8 = FloatRegisterType.from_name("fs8")
    FS9 = FloatRegisterType.from_name("fs9")
    FS10 = FloatRegisterType.from_name("fs10")
    FS11 = FloatRegisterType.from_name("fs11")
    FT8 = FloatRegisterType.from_name("ft8")
    FT9 = FloatRegisterType.from_name("ft9")
    FT10 = FloatRegisterType.from_name("ft10")
    FT11 = FloatRegisterType.from_name("ft11")

    # register classes:

    A = (A0, A1, A2, A3, A4, A5, A6, A7)
    T = (T0, T1, T2, T3, T4, T5, T6)
    S = (S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11)

    FA = (FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7)
    FT = (FT0, FT1, FT2, FT3, FT4, FT5, FT6, FT7, FT8, FT9, FT10, FT11)
    FS = (FS0, FS1, FS2, FS3, FS4, FS5, FS6, FS7, FS8, FS9, FS10, FS11)


def is_non_zero(reg: IntRegisterType) -> bool:
    """
    Returns True if the register is allocated, and is not the x0/ZERO register.
    """
    return (
        reg.is_allocated and not isinstance(reg.index, NoneAttr) and reg.index.data != 0
    )
