from __future__ import annotations

from abc import ABC

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import (
    irdl_attr_definition,
)


class X86RegisterType(RegisterType, ABC):
    """
    The abstract class for all x86 register types.
    """


X86_INDEX_BY_NAME = {
    "rax": 0,
    "rcx": 1,
    "rdx": 2,
    "rbx": 3,
    "rsp": 4,
    "rbp": 5,
    "rsi": 6,
    "rdi": 7,
    # Currently don't support 32-bit registers
    # https://github.com/xdslproject/xdsl/issues/4737
    # "eax": 0,
    # "ecx": 1,
    # "edx": 2,
    # "ebx": 3,
    # "esp": 4,
    # "ebp": 5,
    # "esi": 6,
    # "edi": 7,
    "r8": 8,
    "r9": 9,
    "r10": 10,
    "r11": 11,
    "r12": 12,
    "r13": 13,
    "r14": 14,
    "r15": 15,
}
"""
Mapping of x86 register names to their indices.

See external [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""


@irdl_attr_definition
class GeneralRegisterType(X86RegisterType):
    """
    A scalar x86 register type representing general registers.
    """

    name = "x86.reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return X86_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_reg_"

    @classmethod
    def allocatable_registers(cls):
        """
        Registers of this type that can be used for register allocation.

        Returns a tuple of GeneralRegisterType instances corresponding to registers that
        are allocatable according to the x86-64 System V ABI.
        This excludes registers with special purposes (e.g., stack pointer, base pointer).
        """
        return (
            RAX,
            RCX,
            RDX,
            RBX,
            RSI,
            RDI,
            R8,
            R9,
            R10,
            R11,
            R13,
            R14,
            R15,
        )


UNALLOCATED_GENERAL = GeneralRegisterType.unallocated()
RAX = GeneralRegisterType.from_name("rax")
RCX = GeneralRegisterType.from_name("rcx")
RDX = GeneralRegisterType.from_name("rdx")
RBX = GeneralRegisterType.from_name("rbx")
RSP = GeneralRegisterType.from_name("rsp")
RBP = GeneralRegisterType.from_name("rbp")
RSI = GeneralRegisterType.from_name("rsi")
RDI = GeneralRegisterType.from_name("rdi")

# Currently don't support 32-bit registers
# https://github.com/xdslproject/xdsl/issues/4737
# EAX = GeneralRegisterType.from_name("eax")
# ECX = GeneralRegisterType.from_name("ecx")
# EDX = GeneralRegisterType.from_name("edx")
# EBX = GeneralRegisterType.from_name("ebx")
# ESP = GeneralRegisterType.from_name("esp")
# EBP = GeneralRegisterType.from_name("ebp")
# ESI = GeneralRegisterType.from_name("esi")
# EDI = GeneralRegisterType.from_name("edi")

R8 = GeneralRegisterType.from_name("r8")
R9 = GeneralRegisterType.from_name("r9")
R10 = GeneralRegisterType.from_name("r10")
R11 = GeneralRegisterType.from_name("r11")
R12 = GeneralRegisterType.from_name("r12")
R13 = GeneralRegisterType.from_name("r13")
R14 = GeneralRegisterType.from_name("r14")
R15 = GeneralRegisterType.from_name("r15")

RFLAGS_INDEX_BY_NAME = {
    "rflags": 0,
}


@irdl_attr_definition
class RFLAGSRegisterType(X86RegisterType):
    """
    A scalar x86 register type representing the RFLAGS register.
    """

    name = "x86.rflags"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return RFLAGS_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_rflags_"


UNALLOCATED_RFLAGS = RFLAGSRegisterType.unallocated()
RFLAGS = RFLAGSRegisterType.from_name("rflags")


class X86VectorRegisterType(X86RegisterType):
    """
    The abstract class for all x86 vector register types.
    """


@irdl_attr_definition
class SSERegisterType(X86VectorRegisterType):
    """
    An x86 register type for SSE instructions.
    """

    name = "x86.ssereg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return SSE_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_sse_"


SSE_INDEX_BY_NAME = {
    "xmm0": 0,
    "xmm1": 1,
    "xmm2": 2,
    "xmm3": 3,
    "xmm4": 4,
    "xmm5": 5,
    "xmm6": 6,
    "xmm7": 7,
    "xmm8": 8,
    "xmm9": 9,
    "xmm10": 10,
    "xmm11": 11,
    "xmm12": 12,
    "xmm13": 13,
    "xmm14": 14,
    "xmm15": 15,
}
"""
Mapping of SSE register names to their indices.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

UNALLOCATED_SSE = SSERegisterType.unallocated()
XMM = tuple(SSERegisterType.from_name(f"xmm{i}") for i in range(16))
(
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,
) = XMM


@irdl_attr_definition
class AVX2RegisterType(X86VectorRegisterType):
    """
    An x86 register type for AVX2 instructions.
    """

    name = "x86.avx2reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return AVX2_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_avx2_"

    @classmethod
    def allocatable_registers(cls):
        return YMM


AVX2_INDEX_BY_NAME = {
    "ymm0": 0,
    "ymm1": 1,
    "ymm2": 2,
    "ymm3": 3,
    "ymm4": 4,
    "ymm5": 5,
    "ymm6": 6,
    "ymm7": 7,
    "ymm8": 8,
    "ymm9": 9,
    "ymm10": 10,
    "ymm11": 11,
    "ymm12": 12,
    "ymm13": 13,
    "ymm14": 14,
    "ymm15": 15,
}
"""
Mapping of AVX2 register names to their indices.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

UNALLOCATED_AVX2 = AVX2RegisterType.unallocated()
YMM = tuple(AVX2RegisterType.from_name(f"ymm{i}") for i in range(16))
(
    YMM0,
    YMM1,
    YMM2,
    YMM3,
    YMM4,
    YMM5,
    YMM6,
    YMM7,
    YMM8,
    YMM9,
    YMM10,
    YMM11,
    YMM12,
    YMM13,
    YMM14,
    YMM15,
) = YMM


@irdl_attr_definition
class AVX512RegisterType(X86VectorRegisterType):
    """
    An x86 register type for AVX512 instructions.
    """

    name = "x86.avx512reg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return X86AVX512_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_avx512_"

    @classmethod
    def allocatable_registers(cls):
        return ZMM


X86AVX512_INDEX_BY_NAME = {
    "zmm0": 0,
    "zmm1": 1,
    "zmm2": 2,
    "zmm3": 3,
    "zmm4": 4,
    "zmm5": 5,
    "zmm6": 6,
    "zmm7": 7,
    "zmm8": 8,
    "zmm9": 9,
    "zmm10": 10,
    "zmm11": 11,
    "zmm12": 12,
    "zmm13": 13,
    "zmm14": 14,
    "zmm15": 15,
    "zmm16": 16,
    "zmm17": 17,
    "zmm18": 18,
    "zmm19": 19,
    "zmm20": 20,
    "zmm21": 21,
    "zmm22": 22,
    "zmm23": 23,
    "zmm24": 24,
    "zmm25": 25,
    "zmm26": 26,
    "zmm27": 27,
    "zmm28": 28,
    "zmm29": 29,
    "zmm30": 30,
    "zmm31": 31,
}
"""
Mapping of AVX512 register names to their indices.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

UNALLOCATED_AVX512 = AVX512RegisterType.unallocated()
ZMM = tuple(AVX512RegisterType.from_name(f"zmm{i}") for i in range(32))
(
    ZMM0,
    ZMM1,
    ZMM2,
    ZMM3,
    ZMM4,
    ZMM5,
    ZMM6,
    ZMM7,
    ZMM8,
    ZMM9,
    ZMM10,
    ZMM11,
    ZMM12,
    ZMM13,
    ZMM14,
    ZMM15,
    ZMM16,
    ZMM17,
    ZMM18,
    ZMM19,
    ZMM20,
    ZMM21,
    ZMM22,
    ZMM23,
    ZMM24,
    ZMM25,
    ZMM26,
    ZMM27,
    ZMM28,
    ZMM29,
    ZMM30,
    ZMM31,
) = ZMM


@irdl_attr_definition
class AVX512MaskRegisterType(X86RegisterType):
    """
    An x86 mask register type for AVX512 instructions.
    """

    name = "x86.avx512maskreg"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return X86AVX512_MASK_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_avx512_mask_"

    @classmethod
    def allocatable_registers(cls):
        return K


X86AVX512_MASK_INDEX_BY_NAME = {
    "k0": 0,
    "k1": 1,
    "k2": 2,
    "k3": 3,
    "k4": 4,
    "k5": 5,
    "k6": 6,
    "k7": 7,
}

UNALLOCATED_AVX512_MASK = AVX512MaskRegisterType.unallocated()
K = tuple(AVX512MaskRegisterType.from_name(f"k{i}") for i in range(8))
(
    K0,
    K1,
    K2,
    K3,
    K4,
    K5,
    K6,
    K7,
) = K
