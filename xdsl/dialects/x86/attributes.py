from __future__ import annotations

from xdsl.backend.assembly_printer import RegisterNameSpec
from xdsl.ir import Data, EnumAttribute, SpacedOpaqueSyntaxAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.str_enum import StrEnum


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "x86.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)


B64_NAMES = (
    "rax",
    "rcx",
    "rdx",
    "rbx",
    "rsp",
    "rbp",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
)
"""
64-bit x86 register names.

See external [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

B32_NAMES = (
    "eax",
    "ecx",
    "edx",
    "ebx",
    "esp",
    "ebp",
    "esi",
    "edi",
    "r8d",
    "r9d",
    "r10d",
    "r11d",
    "r12d",
    "r13d",
    "r14d",
    "r15d",
)
"""
32-bit x86 register names.

See external [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

B16_NAMES = (
    "ax",
    "cx",
    "dx",
    "bx",
    "sp",
    "bp",
    "si",
    "di",
    "r8w",
    "r9w",
    "r10w",
    "r11w",
    "r12w",
    "r13w",
    "r14w",
    "r15w",
)
"""
16-bit x86 register names.

See external [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""


class GeneralPurposeRegisterWidth(StrEnum):
    """
    Supported x86 general-purpose register sizes:

      - B64 → 64-bit (rax, rcx, ..., r8, r9, ...)
      - B32 → 32-bit (eax, ecx, ..., r8d, r9d, ...)
      - B16 → 16-bit (ax, cx, ..., r8w, r9w, ...)
    """

    B64 = "b64"
    B32 = "b32"
    B16 = "b16"
    # Don't support 8-bit for now

    @property
    def bitwidth(self):
        return _GPR_BITWIDTH_BY_SIZE[self]

    @staticmethod
    def from_bitwidth(bitwidth: int):
        try:
            return _GPR_SIZE_BY_BITWIDTH[bitwidth]
        except KeyError:
            raise ValueError(
                f"No general-purpose register size for bitwidth {bitwidth}."
            )


_GP_NAME_BY_INDEX_BY_BITWIDTH = {
    GeneralPurposeRegisterWidth.B64: B64_NAMES,
    GeneralPurposeRegisterWidth.B32: B32_NAMES,
    GeneralPurposeRegisterWidth.B16: B16_NAMES,
}
_GPR_BITWIDTH_BY_SIZE = {
    GeneralPurposeRegisterWidth.B64: 64,
    GeneralPurposeRegisterWidth.B32: 32,
    GeneralPurposeRegisterWidth.B16: 16,
}
_GPR_SIZE_BY_BITWIDTH = {v: k for k, v in _GPR_BITWIDTH_BY_SIZE.items()}


@irdl_attr_definition
class GeneralPurposeRegisterWidthAttr(
    EnumAttribute[GeneralPurposeRegisterWidth],
    SpacedOpaqueSyntaxAttribute,
    RegisterNameSpec,
):
    """
    Attribute containing the general-purpose register size specification.
    """

    name = "x86.gpr_reg_size"

    def get_register_name(self, index: int) -> str:
        return _GP_NAME_BY_INDEX_BY_BITWIDTH[self.data][index]


B64 = GeneralPurposeRegisterWidthAttr(GeneralPurposeRegisterWidth.B64)
B32 = GeneralPurposeRegisterWidthAttr(GeneralPurposeRegisterWidth.B32)
B16 = GeneralPurposeRegisterWidthAttr(GeneralPurposeRegisterWidth.B16)

SSE_NAMES = tuple(f"xmm{i}" for i in range(16))
"""
Valid SSE register names.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

AVX2_NAMES = tuple(f"ymm{i}" for i in range(16))
"""
Valid AVX2 register names.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""

AVX512_NAMES = tuple(f"zmm{i}" for i in range(32))
"""
Valid AVX512 register names.

See external # [documentation](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers).
"""


class VectorRegisterWidth(StrEnum):
    """
    One of three supported x86 vector sizes:

      - B128 → 128-bit SSE Vector  (XMM)
      - B256 → 256-bit AVX Vector  (YMM)
      - B512 → 512-bit AVX Vector  (ZMM)
    """

    B128 = "b128"
    B256 = "b256"
    B512 = "b512"

    @property
    def bitwidth(self):
        return _VECTOR_BITWIDTH_BY_SIZE[self]

    @staticmethod
    def from_bitwidth(bitwidth: int):
        try:
            return _VECTOR_SIZE_BY_BITWIDTH[bitwidth]
        except KeyError:
            raise ValueError(f"No vector register size for bitwidth {bitwidth}.")


_VECTOR_BITWIDTH_BY_SIZE = {
    VectorRegisterWidth.B128: 128,
    VectorRegisterWidth.B256: 256,
    VectorRegisterWidth.B512: 512,
}
_VECTOR_SIZE_BY_BITWIDTH = {v: k for k, v in _VECTOR_BITWIDTH_BY_SIZE.items()}

_VECTOR_NAME_BY_INDEX_BY_BITWIDTH = {
    VectorRegisterWidth.B128: SSE_NAMES,
    VectorRegisterWidth.B256: AVX2_NAMES,
    VectorRegisterWidth.B512: AVX512_NAMES,
}


@irdl_attr_definition
class VectorRegisterWidthAttr(
    EnumAttribute[VectorRegisterWidth], SpacedOpaqueSyntaxAttribute, RegisterNameSpec
):
    """
    Attribute containing the vector register size specification.
    """

    name = "x86.vec_reg_size"

    def get_register_name(self, index: int) -> str:
        return _VECTOR_NAME_BY_INDEX_BY_BITWIDTH[self.data][index]


B128 = VectorRegisterWidthAttr(VectorRegisterWidth.B128)
B256 = VectorRegisterWidthAttr(VectorRegisterWidth.B256)
B512 = VectorRegisterWidthAttr(VectorRegisterWidth.B512)
