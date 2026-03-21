from __future__ import annotations

from abc import ABC

from typing_extensions import override

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import (
    irdl_attr_definition,
)


class X86RegisterType(RegisterType, ABC):
    """
    The abstract class for all x86 register types.
    """


X86_GPR_POOL_KEY = "x86.reg"
"""
Pool key for x86 general-purpose register allocation.

64-bit, 32-bit, 16-bit, and 8-bit names share one physical register per index (e.g.
``rax`` / ``eax`` / ``ax`` / ``al``). Indices are drawn from a single pool; see concrete
subclasses of [xdsl.dialects.x86.GeneralRegisterType].
"""


_ALLOCATABLE_GPR_INDICES = frozenset({0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 13, 14, 15})
"""
Indices allocatable per x86-64 System V ABI (excludes rsp, rbp, r12 in this dialect).
"""


class GeneralRegisterType(X86RegisterType, ABC):
    """
    Abstract base for x86 scalar general-purpose register types (64/32/16/8-bit names).

    Concrete subclasses set the assembly name width; all share :data:`X86_GPR_POOL_KEY`
    for register allocation.
    """

    @classmethod
    @override
    def register_pool_key(cls) -> str:
        return X86_GPR_POOL_KEY

    @classmethod
    def allocatable_registers(cls):
        """
        Registers of this type that can be used for register allocation.

        Returns a tuple of GeneralRegisterType instances corresponding to registers that
        are allocatable according to the x86-64 System V ABI.
        This excludes registers with special purposes (e.g., stack pointer, base pointer).
        """
        return tuple(cls.from_index(i) for i in _ALLOCATABLE_GPR_INDICES)


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
"""
Mapping of x86 64-bit register names to their indices.

See external [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

REG32_INDEX_BY_NAME = {
    "eax": 0,
    "ecx": 1,
    "edx": 2,
    "ebx": 3,
    "esp": 4,
    "ebp": 5,
    "esi": 6,
    "edi": 7,
    "r8d": 8,
    "r9d": 9,
    "r10d": 10,
    "r11d": 11,
    "r12d": 12,
    "r13d": 13,
    "r14d": 14,
    "r15d": 15,
}

REG16_INDEX_BY_NAME = {
    "ax": 0,
    "cx": 1,
    "dx": 2,
    "bx": 3,
    "sp": 4,
    "bp": 5,
    "si": 6,
    "di": 7,
    "r8w": 8,
    "r9w": 9,
    "r10w": 10,
    "r11w": 11,
    "r12w": 12,
    "r13w": 13,
    "r14w": 14,
    "r15w": 15,
}

REG8_INDEX_BY_NAME = {
    "al": 0,
    "cl": 1,
    "dl": 2,
    "bl": 3,
    "spl": 4,
    "bpl": 5,
    "sil": 6,
    "dil": 7,
    "r8b": 8,
    "r9b": 9,
    "r10b": 10,
    "r11b": 11,
    "r12b": 12,
    "r13b": 13,
    "r14b": 14,
    "r15b": 15,
}
"""
Low-byte 8-bit names only; ``ah``/``ch``/``dh``/``bh`` are not modeled (see
:class:`Reg8Type`).
"""


@irdl_attr_definition
class Reg64Type(GeneralRegisterType):
    """
    64-bit x86 general-purpose register names (``rax``, ``r8``, …).
    """

    name = "x86.reg64"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return X86_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_reg_"


@irdl_attr_definition
class Reg32Type(GeneralRegisterType):
    """
    32-bit x86 general-purpose register names (``eax``, ``r8d``, …).
    """

    name = "x86.reg32"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return REG32_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_reg32_"


@irdl_attr_definition
class Reg16Type(GeneralRegisterType):
    """
    16-bit x86 general-purpose register names (``ax``, ``r8w``, …).
    """

    name = "x86.reg16"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return REG16_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_reg16_"


@irdl_attr_definition
class Reg8Type(GeneralRegisterType):
    """
    8-bit x86 low-byte register names (``al``, ``r8b``, …).

    High-byte names ``ah``/``ch``/``dh``/``bh`` are not represented: they share indices
    with low bytes but :class:`~xdsl.backend.register_type.RegisterType` assumes one
    canonical name per pool index.
    """

    name = "x86.reg8"

    @classmethod
    def index_by_name(cls) -> dict[str, int]:
        return REG8_INDEX_BY_NAME

    @classmethod
    def infinite_register_prefix(cls):
        return "inf_reg8_"


UNALLOCATED_REG64 = Reg64Type.unallocated()
UNALLOCATED_REG32 = Reg32Type.unallocated()
UNALLOCATED_REG16 = Reg16Type.unallocated()
UNALLOCATED_REG8 = Reg8Type.unallocated()

RAX = Reg64Type.from_name("rax")
RCX = Reg64Type.from_name("rcx")
RDX = Reg64Type.from_name("rdx")
RBX = Reg64Type.from_name("rbx")
RSP = Reg64Type.from_name("rsp")
RBP = Reg64Type.from_name("rbp")
RSI = Reg64Type.from_name("rsi")
RDI = Reg64Type.from_name("rdi")
R8 = Reg64Type.from_name("r8")
R9 = Reg64Type.from_name("r9")
R10 = Reg64Type.from_name("r10")
R11 = Reg64Type.from_name("r11")
R12 = Reg64Type.from_name("r12")
R13 = Reg64Type.from_name("r13")
R14 = Reg64Type.from_name("r14")
R15 = Reg64Type.from_name("r15")

EAX = Reg32Type.from_name("eax")
ECX = Reg32Type.from_name("ecx")
EDX = Reg32Type.from_name("edx")
EBX = Reg32Type.from_name("ebx")
ESP = Reg32Type.from_name("esp")
EBP = Reg32Type.from_name("ebp")
ESI = Reg32Type.from_name("esi")
EDI = Reg32Type.from_name("edi")
R8D = Reg32Type.from_name("r8d")
R9D = Reg32Type.from_name("r9d")
R10D = Reg32Type.from_name("r10d")
R11D = Reg32Type.from_name("r11d")
R12D = Reg32Type.from_name("r12d")
R13D = Reg32Type.from_name("r13d")
R14D = Reg32Type.from_name("r14d")
R15D = Reg32Type.from_name("r15d")

AX = Reg16Type.from_name("ax")
CX = Reg16Type.from_name("cx")
DX = Reg16Type.from_name("dx")
BX = Reg16Type.from_name("bx")
SP = Reg16Type.from_name("sp")
BP = Reg16Type.from_name("bp")
SI = Reg16Type.from_name("si")
DI = Reg16Type.from_name("di")
R8W = Reg16Type.from_name("r8w")
R9W = Reg16Type.from_name("r9w")
R10W = Reg16Type.from_name("r10w")
R11W = Reg16Type.from_name("r11w")
R12W = Reg16Type.from_name("r12w")
R13W = Reg16Type.from_name("r13w")
R14W = Reg16Type.from_name("r14w")
R15W = Reg16Type.from_name("r15w")

AL = Reg8Type.from_name("al")
CL = Reg8Type.from_name("cl")
DL = Reg8Type.from_name("dl")
BL = Reg8Type.from_name("bl")
SPL = Reg8Type.from_name("spl")
BPL = Reg8Type.from_name("bpl")
SIL = Reg8Type.from_name("sil")
DIL = Reg8Type.from_name("dil")
R8B = Reg8Type.from_name("r8b")
R9B = Reg8Type.from_name("r9b")
R10B = Reg8Type.from_name("r10b")
R11B = Reg8Type.from_name("r11b")
R12B = Reg8Type.from_name("r12b")
R13B = Reg8Type.from_name("r13b")
R14B = Reg8Type.from_name("r14b")
R15B = Reg8Type.from_name("r15b")

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


X86_VECTOR_POOL_KEY = "x86.vector"
"""
Pool key for x86 vector register allocation.

``xmm*``, ``ymm*``, and ``zmm*`` share one physical register per index where they
overlap (e.g. xmm0/ymm0/zmm0). Indices are drawn from a single pool; targets configure
how many names exist (e.g. 8, 16, or 32) via :meth:`allocatable_registers`.
"""


class X86VectorRegisterType(X86RegisterType):
    """
    The abstract class for all x86 vector register types.
    """

    @override
    @classmethod
    def register_pool_key(cls) -> str:
        return X86_VECTOR_POOL_KEY


SSE_INDEX_BY_NAME = {f"xmm{i}": i for i in range(32)}
"""
Mapping of SSE register names to their indices.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
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

    @classmethod
    def allocatable_registers(cls):
        return XMM


UNALLOCATED_SSE = SSERegisterType.unallocated()
XMM = tuple(SSERegisterType.from_name(f"xmm{i}") for i in range(32))
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
    XMM16,
    XMM17,
    XMM18,
    XMM19,
    XMM20,
    XMM21,
    XMM22,
    XMM23,
    XMM24,
    XMM25,
    XMM26,
    XMM27,
    XMM28,
    XMM29,
    XMM30,
    XMM31,
) = XMM


AVX2_INDEX_BY_NAME = {f"ymm{i}": i for i in range(32)}
"""
Mapping of AVX2 register names to their indices.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""


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


UNALLOCATED_AVX2 = AVX2RegisterType.unallocated()
YMM = tuple(AVX2RegisterType.from_name(f"ymm{i}") for i in range(32))
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
    YMM16,
    YMM17,
    YMM18,
    YMM19,
    YMM20,
    YMM21,
    YMM22,
    YMM23,
    YMM24,
    YMM25,
    YMM26,
    YMM27,
    YMM28,
    YMM29,
    YMM30,
    YMM31,
) = YMM


X86AVX512_INDEX_BY_NAME = {f"zmm{i}": i for i in range(32)}
"""
Mapping of AVX512 register names to their indices.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""


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


X86AVX512_MASK_INDEX_BY_NAME = {f"k{i}": i for i in range(8)}


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
