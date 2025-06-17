"""
The math_xdsl dialect contains extensions to the math dialect.
"""

from enum import auto

from xdsl.ir import Attribute, Dialect, EnumAttribute, SpacedOpaqueSyntaxAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.str_enum import StrEnum


class Constant(StrEnum):
    E = auto()  # 𝑒
    PI = auto()  # π
    M_2_SQRTPI = auto()  # 2/sqrt(π)
    LOG2E = auto()  # log2(𝑒)
    PI_2 = auto()  # π/2
    SQRT2 = auto()  # sqrt(2)
    LOG10E = auto()  # log10(𝑒)
    PI_4 = auto()  # π/4
    SQRT1_2 = auto()  # sqrt(1/2)
    LN2 = auto()  # ln(2)
    M_1_PI = auto()  # 1/π
    INFINITY = auto()  # ∞
    LN10 = auto()  # ln(10)
    M_2_PI = auto()  # 2/π


@irdl_attr_definition
class ConstantAttr(EnumAttribute[Constant], SpacedOpaqueSyntaxAttribute):
    name = "math_xdsl.constant"


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "math_xdsl.constant"

    symbol = prop_def(ConstantAttr)

    value = result_def()

    def __init__(self, symbol: Constant | ConstantAttr, result_type: Attribute):
        if not isinstance(symbol, ConstantAttr):
            symbol = ConstantAttr(symbol)
        super().__init__(
            properties={
                "symbol": symbol,
            },
            result_types=[result_type],
        )

    @classmethod
    def parse(cls, parser: Parser) -> "ConstantOp":
        symbol_str = parser.parse_identifier()
        symbol = ConstantAttr(Constant(symbol_str))

        parser.parse_punctuation(":")
        result_type = parser.parse_type()

        return ConstantOp(symbol, result_type)

    def print(self, printer: Printer):
        printer.print_string(f" {self.symbol.data} : ")
        printer.print_attribute(self.value.type)


MathXDSL = Dialect(
    "math_xdsl",
    [
        ConstantOp,
    ],
    [
        ConstantAttr,
    ],
)
