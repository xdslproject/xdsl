from __future__ import annotations

from collections.abc import Sequence

from xdsl.backend.register_type import RegisterType
from xdsl.ir import Attribute
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.utils.exceptions import VerifyException


class X86RegisterType(RegisterType):
    """
    The abstract class for all x86 register types.
    """

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        if parser.parse_optional_punctuation("<") is not None:
            name = parser.parse_identifier()
            parser.parse_punctuation(">")
            if not name.startswith("e") and not name.startswith("r"):
                assert name in cls.abi_index_by_name(), f"{name}"
        else:
            name = ""
        return cls._parameters_from_spelling(name)

    def verify(self) -> None:
        name = self.spelling.data
        if not self.is_allocated or name.startswith("e") or name.startswith("r"):
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")


X86_INDEX_BY_NAME = {
    "rax": 0,
    "rcx": 1,
    "rdx": 2,
    "rbx": 3,
    "rsp": 4,
    "rbp": 5,
    "rsi": 6,
    "rdi": 7,
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
    def unallocated(cls) -> GeneralRegisterType:
        return UNALLOCATED_GENERAL

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return X86_INDEX_BY_NAME


UNALLOCATED_GENERAL = GeneralRegisterType("")
RAX = GeneralRegisterType("rax")
RCX = GeneralRegisterType("rcx")
RDX = GeneralRegisterType("rdx")
RBX = GeneralRegisterType("rbx")
RSP = GeneralRegisterType("rsp")
RBP = GeneralRegisterType("rbp")
RSI = GeneralRegisterType("rsi")
RDI = GeneralRegisterType("rdi")
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
    def unallocated(cls) -> RFLAGSRegisterType:
        return UNALLOCATED_RFLAGS

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return RFLAGS_INDEX_BY_NAME


UNALLOCATED_RFLAGS = RFLAGSRegisterType("")
RFLAGS = RFLAGSRegisterType("rflags")


class X86VectorRegisterType(X86RegisterType):
    pass


@irdl_attr_definition
class AVX512RegisterType(X86VectorRegisterType):
    """
    An x86 register type for AVX512 instructions.
    """

    name = "x86.avx512reg"

    @classmethod
    def unallocated(cls) -> AVX512RegisterType:
        return UNALLOCATED_AVX512

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86AVX512"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return X86AVX512_INDEX_BY_NAME


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
