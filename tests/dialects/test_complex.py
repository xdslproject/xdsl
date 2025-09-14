from xdsl.dialects import complex
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    i32,
)
from xdsl.traits import ConstantLike


def test_constant_construction():
    c1 = complex.ConstantOp(
        value=ArrayAttr([IntAttr(42), IntAttr(43)]),
        result_type=complex.ComplexType(i32),
    )
    constantlike = c1.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(c1) == ArrayAttr([IntAttr(42), IntAttr(43)])
