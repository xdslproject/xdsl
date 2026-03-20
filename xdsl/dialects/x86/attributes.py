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
        return _BITWIDTH_BY_SIZE[self]

    @staticmethod
    def from_bitwidth(bitwidth: int):
        try:
            return _SIZE_BY_BITWIDTH[bitwidth]
        except KeyError:
            raise ValueError(f"No vector register size for bitwidth {bitwidth}.")


_BITWIDTH_BY_SIZE = {
    VectorRegisterWidth.B128: 128,
    VectorRegisterWidth.B256: 256,
    VectorRegisterWidth.B512: 512,
}
_SIZE_BY_BITWIDTH = {v: k for k, v in _BITWIDTH_BY_SIZE.items()}

_NAME_BY_INDEX_BY_BITWIDTH = {
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
        return _NAME_BY_INDEX_BY_BITWIDTH[self.data][index]


B128 = VectorRegisterWidthAttr(VectorRegisterWidth.B128)
B256 = VectorRegisterWidthAttr(VectorRegisterWidth.B256)
B512 = VectorRegisterWidthAttr(VectorRegisterWidth.B512)
