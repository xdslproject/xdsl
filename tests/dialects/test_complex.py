from xdsl.dialects import complex
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    i32,
)


def test_constant_construction():
    c1 = complex.ConstantOp(
        value=ArrayAttr([IntAttr(42), IntAttr(43)]),
        result_type=complex.ComplexType(i32),
    )
    assert c1.get_constant_value() == ArrayAttr([IntAttr(42), IntAttr(43)])
