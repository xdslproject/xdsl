from __future__ import annotations

from abc import ABC

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.utils.exceptions import VerifyException


class X86RegisterType(RegisterType, ABC):
    """
    The abstract class for all x86 register types.
    """

    def verify(self) -> None:
        name = self.spelling.data
        if not self.is_allocated:
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")


# See https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers
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


@irdl_attr_definition
class GeneralRegisterType(X86RegisterType):
    """
    A scalar x86 register type representing general registers.
    """

    name = "x86.reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return X86_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_reg_"


UNALLOCATED_GENERAL = GeneralRegisterType("")
RAX = GeneralRegisterType("rax")
RCX = GeneralRegisterType("rcx")
RDX = GeneralRegisterType("rdx")
RBX = GeneralRegisterType("rbx")
RSP = GeneralRegisterType("rsp")
RBP = GeneralRegisterType("rbp")
RSI = GeneralRegisterType("rsi")
RDI = GeneralRegisterType("rdi")

EAX = GeneralRegisterType("eax")
ECX = GeneralRegisterType("ecx")
EDX = GeneralRegisterType("edx")
EBX = GeneralRegisterType("ebx")
ESP = GeneralRegisterType("esp")
EBP = GeneralRegisterType("ebp")
ESI = GeneralRegisterType("esi")
EDI = GeneralRegisterType("edi")

R8 = GeneralRegisterType("r8")
R9 = GeneralRegisterType("r9")
R10 = GeneralRegisterType("r10")
R11 = GeneralRegisterType("r11")
R12 = GeneralRegisterType("r12")
R13 = GeneralRegisterType("r13")
R14 = GeneralRegisterType("r14")
R15 = GeneralRegisterType("r15")

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
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return RFLAGS_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_rflags_"


UNALLOCATED_RFLAGS = RFLAGSRegisterType("")
RFLAGS = RFLAGSRegisterType("rflags")


class X86VectorRegisterType(X86RegisterType):
    """
    The abstract class for all x86 vector register types.
    """

    pass


@irdl_attr_definition
class SSERegisterType(X86VectorRegisterType):
    """
    An x86 register type for SSE instructions.
    """

    name = "x86.ssereg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "SSE"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return SSE_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_sse_"


# See https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers
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

UNALLOCATED_SSE = SSERegisterType("")
XMM0 = SSERegisterType("xmm0")
XMM1 = SSERegisterType("xmm1")
XMM2 = SSERegisterType("xmm2")
XMM3 = SSERegisterType("xmm3")
XMM4 = SSERegisterType("xmm4")
XMM5 = SSERegisterType("xmm5")
XMM6 = SSERegisterType("xmm6")
XMM7 = SSERegisterType("xmm7")
XMM8 = SSERegisterType("xmm8")
XMM9 = SSERegisterType("xmm9")
XMM10 = SSERegisterType("xmm10")
XMM11 = SSERegisterType("xmm11")
XMM12 = SSERegisterType("xmm12")
XMM13 = SSERegisterType("xmm13")
XMM14 = SSERegisterType("xmm14")
XMM15 = SSERegisterType("xmm15")


@irdl_attr_definition
class AVX2RegisterType(X86VectorRegisterType):
    """
    An x86 register type for AVX2 instructions.
    """

    name = "x86.avx2reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "AVX2"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return AVX2_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_avx2_"


# See https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers
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

UNALLOCATED_AVX2 = AVX2RegisterType("")
YMM0 = AVX2RegisterType("ymm0")
YMM1 = AVX2RegisterType("ymm1")
YMM2 = AVX2RegisterType("ymm2")
YMM3 = AVX2RegisterType("ymm3")
YMM4 = AVX2RegisterType("ymm4")
YMM5 = AVX2RegisterType("ymm5")
YMM6 = AVX2RegisterType("ymm6")
YMM7 = AVX2RegisterType("ymm7")
YMM8 = AVX2RegisterType("ymm8")
YMM9 = AVX2RegisterType("ymm9")
YMM10 = AVX2RegisterType("ymm10")
YMM11 = AVX2RegisterType("ymm11")
YMM12 = AVX2RegisterType("ymm12")
YMM13 = AVX2RegisterType("ymm13")
YMM14 = AVX2RegisterType("ymm14")
YMM15 = AVX2RegisterType("ymm15")


@irdl_attr_definition
class AVX512RegisterType(X86VectorRegisterType):
    """
    An x86 register type for AVX512 instructions.
    """

    name = "x86.avx512reg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "AVX512"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return X86AVX512_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_avx512_"


# See https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers
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

UNALLOCATED_AVX512 = AVX512RegisterType("")
ZMM0 = AVX512RegisterType("zmm0")
ZMM1 = AVX512RegisterType("zmm1")
ZMM2 = AVX512RegisterType("zmm2")
ZMM3 = AVX512RegisterType("zmm3")
ZMM4 = AVX512RegisterType("zmm4")
ZMM5 = AVX512RegisterType("zmm5")
ZMM6 = AVX512RegisterType("zmm6")
ZMM7 = AVX512RegisterType("zmm7")
ZMM8 = AVX512RegisterType("zmm8")
ZMM9 = AVX512RegisterType("zmm9")
ZMM10 = AVX512RegisterType("zmm10")
ZMM11 = AVX512RegisterType("zmm11")
ZMM12 = AVX512RegisterType("zmm12")
ZMM13 = AVX512RegisterType("zmm13")
ZMM14 = AVX512RegisterType("zmm14")
ZMM15 = AVX512RegisterType("zmm15")
ZMM16 = AVX512RegisterType("zmm16")
ZMM17 = AVX512RegisterType("zmm17")
ZMM18 = AVX512RegisterType("zmm18")
ZMM19 = AVX512RegisterType("zmm19")
ZMM20 = AVX512RegisterType("zmm20")
ZMM21 = AVX512RegisterType("zmm21")
ZMM22 = AVX512RegisterType("zmm22")
ZMM23 = AVX512RegisterType("zmm23")
ZMM24 = AVX512RegisterType("zmm24")
ZMM25 = AVX512RegisterType("zmm25")
ZMM26 = AVX512RegisterType("zmm26")
ZMM27 = AVX512RegisterType("zmm27")
ZMM28 = AVX512RegisterType("zmm28")
ZMM29 = AVX512RegisterType("zmm29")
ZMM30 = AVX512RegisterType("zmm30")
ZMM31 = AVX512RegisterType("zmm31")
