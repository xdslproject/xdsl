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
    "eax": 0,
    "ecx": 1,
    "edx": 2,
    "ebx": 3,
    "esp": 4,
    "ebp": 5,
    "esi": 6,
    "edi": 7,
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


UNALLOCATED_GENERAL = GeneralRegisterType.unallocated()
RAX = GeneralRegisterType.from_name("rax")
RCX = GeneralRegisterType.from_name("rcx")
RDX = GeneralRegisterType.from_name("rdx")
RBX = GeneralRegisterType.from_name("rbx")
RSP = GeneralRegisterType.from_name("rsp")
RBP = GeneralRegisterType.from_name("rbp")
RSI = GeneralRegisterType.from_name("rsi")
RDI = GeneralRegisterType.from_name("rdi")

EAX = GeneralRegisterType.from_name("eax")
ECX = GeneralRegisterType.from_name("ecx")
EDX = GeneralRegisterType.from_name("edx")
EBX = GeneralRegisterType.from_name("ebx")
ESP = GeneralRegisterType.from_name("esp")
EBP = GeneralRegisterType.from_name("ebp")
ESI = GeneralRegisterType.from_name("esi")
EDI = GeneralRegisterType.from_name("edi")

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
XMM0 = SSERegisterType.from_name("xmm0")
XMM1 = SSERegisterType.from_name("xmm1")
XMM2 = SSERegisterType.from_name("xmm2")
XMM3 = SSERegisterType.from_name("xmm3")
XMM4 = SSERegisterType.from_name("xmm4")
XMM5 = SSERegisterType.from_name("xmm5")
XMM6 = SSERegisterType.from_name("xmm6")
XMM7 = SSERegisterType.from_name("xmm7")
XMM8 = SSERegisterType.from_name("xmm8")
XMM9 = SSERegisterType.from_name("xmm9")
XMM10 = SSERegisterType.from_name("xmm10")
XMM11 = SSERegisterType.from_name("xmm11")
XMM12 = SSERegisterType.from_name("xmm12")
XMM13 = SSERegisterType.from_name("xmm13")
XMM14 = SSERegisterType.from_name("xmm14")
XMM15 = SSERegisterType.from_name("xmm15")


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
YMM0 = AVX2RegisterType.from_name("ymm0")
YMM1 = AVX2RegisterType.from_name("ymm1")
YMM2 = AVX2RegisterType.from_name("ymm2")
YMM3 = AVX2RegisterType.from_name("ymm3")
YMM4 = AVX2RegisterType.from_name("ymm4")
YMM5 = AVX2RegisterType.from_name("ymm5")
YMM6 = AVX2RegisterType.from_name("ymm6")
YMM7 = AVX2RegisterType.from_name("ymm7")
YMM8 = AVX2RegisterType.from_name("ymm8")
YMM9 = AVX2RegisterType.from_name("ymm9")
YMM10 = AVX2RegisterType.from_name("ymm10")
YMM11 = AVX2RegisterType.from_name("ymm11")
YMM12 = AVX2RegisterType.from_name("ymm12")
YMM13 = AVX2RegisterType.from_name("ymm13")
YMM14 = AVX2RegisterType.from_name("ymm14")
YMM15 = AVX2RegisterType.from_name("ymm15")


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
ZMM0 = AVX512RegisterType.from_name("zmm0")
ZMM1 = AVX512RegisterType.from_name("zmm1")
ZMM2 = AVX512RegisterType.from_name("zmm2")
ZMM3 = AVX512RegisterType.from_name("zmm3")
ZMM4 = AVX512RegisterType.from_name("zmm4")
ZMM5 = AVX512RegisterType.from_name("zmm5")
ZMM6 = AVX512RegisterType.from_name("zmm6")
ZMM7 = AVX512RegisterType.from_name("zmm7")
ZMM8 = AVX512RegisterType.from_name("zmm8")
ZMM9 = AVX512RegisterType.from_name("zmm9")
ZMM10 = AVX512RegisterType.from_name("zmm10")
ZMM11 = AVX512RegisterType.from_name("zmm11")
ZMM12 = AVX512RegisterType.from_name("zmm12")
ZMM13 = AVX512RegisterType.from_name("zmm13")
ZMM14 = AVX512RegisterType.from_name("zmm14")
ZMM15 = AVX512RegisterType.from_name("zmm15")
ZMM16 = AVX512RegisterType.from_name("zmm16")
ZMM17 = AVX512RegisterType.from_name("zmm17")
ZMM18 = AVX512RegisterType.from_name("zmm18")
ZMM19 = AVX512RegisterType.from_name("zmm19")
ZMM20 = AVX512RegisterType.from_name("zmm20")
ZMM21 = AVX512RegisterType.from_name("zmm21")
ZMM22 = AVX512RegisterType.from_name("zmm22")
ZMM23 = AVX512RegisterType.from_name("zmm23")
ZMM24 = AVX512RegisterType.from_name("zmm24")
ZMM25 = AVX512RegisterType.from_name("zmm25")
ZMM26 = AVX512RegisterType.from_name("zmm26")
ZMM27 = AVX512RegisterType.from_name("zmm27")
ZMM28 = AVX512RegisterType.from_name("zmm28")
ZMM29 = AVX512RegisterType.from_name("zmm29")
ZMM30 = AVX512RegisterType.from_name("zmm30")
ZMM31 = AVX512RegisterType.from_name("zmm31")
