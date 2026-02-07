import pytest

from xdsl.dialects import complex
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    ComplexType,
    FloatAttr,
    FloatData,
    IntAttr,
    IntegerAttr,
    IntegerType,
    f16,
    f32,
    f64,
    i1,
    i8,
    i16,
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


class Test_constant_op_helper_constructors:
    @pytest.mark.parametrize(
        "real, imag, type",
        [
            (0, 1, i1),
            (1, 2, i8),
            (24, 19, i16),
            (33, 88, i32),
        ],
    )
    def test_from_ints(self, real: int, imag: int, type: IntegerType):
        cst = complex.ConstantOp.from_ints((real, imag), type)
        assert cst.value.data[0].type == type
        assert cst.value == ArrayAttr(
            [IntegerAttr(real, type), IntegerAttr(imag, type)]
        )
        assert cst.result_types[0] == ComplexType(type)

    @pytest.mark.parametrize(
        "real, imag, type",
        [
            (0, 1, f16),
            (-1, 2.2, f32),
            (24.45, 19.0, f64),
        ],
    )
    def test_from_floats(self, real: float, imag: float, type: AnyFloat):
        cst = complex.ConstantOp.from_floats((real, imag), type)
        assert cst.value.data[0].type == type
        assert cst.value == ArrayAttr([FloatAttr(real, type), FloatAttr(imag, type)])
        assert cst.result_types[0] == ComplexType(type)


@pytest.mark.parametrize(
    "real, imag, type",
    [
        (2.1, -20, f16),
        (-1.2, 2.5, f32),
        (3, 1, f64),
    ],
)
def test_complex_number_attr(real: float, imag: float, type: AnyFloat):
    attr1 = complex.ComplexNumberAttr(real, imag, ComplexType(type))
    attr2 = complex.ComplexNumberAttr[type](
        FloatData(real), FloatData(imag), ComplexType(type)
    )
    assert attr1.real == attr2.real
    assert attr1.imag == attr2.imag
    assert attr1.type == attr2.type
