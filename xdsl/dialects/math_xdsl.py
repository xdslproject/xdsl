"""
The math_xdsl dialect contains extensions to the math dialect.
Currently, it only contains the `math_xdsl.constant` operation and
a related attribute, which can be used to define and use
mathematical constants in IR.

The idea is to upstream these extension to the math dialect in MLIR
at some point when they have matured.
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
from xdsl.utils.str_enum import StrEnum


class Constant(StrEnum):
    E = auto()  # e
    PI = auto()  # π
    M_2_SQRTPI = auto()  # 2/sqrt(π)
    LOG2E = auto()  # log2(e)
    PI_2 = auto()  # π/2
    SQRT2 = auto()  # sqrt(2)
    LOG10E = auto()  # log10(e)
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

    assembly_format = "$symbol attr-dict `:` type($value)"

    def __init__(self, symbol: Constant | ConstantAttr, result_type: Attribute):
        if not isinstance(symbol, ConstantAttr):
            symbol = ConstantAttr(symbol)
        super().__init__(
            properties={
                "symbol": symbol,
            },
            result_types=[result_type],
        )


MathXDSL = Dialect(
    "math_xdsl",
    [
        ConstantOp,
    ],
    [
        ConstantAttr,
    ],
)
