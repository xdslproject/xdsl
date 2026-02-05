from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

from xdsl.dialects.builtin import IntegerType, Signedness
from xdsl.dialects.utils import FastMathAttrBase, FastMathFlag
from xdsl.ir import Data
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class FastMathFlagsAttr(FastMathAttrBase):
    """
    riscv.fastmath is a mirror of LLVMs fastmath flags.
    """

    name = "riscv.fastmath"

    def __init__(self, flags: None | Sequence[FastMathFlag] | Literal["none", "fast"]):
        # irdl_attr_definition defines an __init__ if none is defined, so we need to
        # explicitely define one here.
        super().__init__(flags)


UI5: TypeAlias = IntegerType[Literal[5], Literal[Signedness.UNSIGNED]]
SI20: TypeAlias = IntegerType[Literal[20], Literal[Signedness.SIGNED]]
SI12: TypeAlias = IntegerType[Literal[12], Literal[Signedness.SIGNED]]
I12: TypeAlias = IntegerType[Literal[12], Literal[Signedness.SIGNLESS]]
I20: TypeAlias = IntegerType[Literal[20], Literal[Signedness.SIGNLESS]]
ui5: UI5 = IntegerType(5, Signedness.UNSIGNED)
si20: SI20 = IntegerType(20, Signedness.SIGNED)
si12: SI12 = IntegerType(12, Signedness.SIGNED)
i12: I12 = IntegerType(12, Signedness.SIGNLESS)
i20: I20 = IntegerType(20, Signedness.SIGNLESS)


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "riscv.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)
