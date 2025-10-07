import pytest

from xdsl.dialects import complex
from xdsl.dialects.arith import (
    FastMathFlagsAttr,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    ComplexType,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    f16,
    f32,
    f64,
    f80,
    f128,
    i1,
    i16,
    i32,
)
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_constant_construction():
    c1 = complex.ConstantOp(
        value=ArrayAttr([IntAttr(42), IntAttr(43)]),
        result_type=complex.ComplexType(i32),
    )
    constantlike = c1.get_trait(ConstantLike)
    assert constantlike is not None
    assert constantlike.get_constant_value(c1) == ArrayAttr([IntAttr(42), IntAttr(43)])


@pytest.mark.parametrize(
    "tuple, width, result_type, value_type",
    [
        (
            (1, 0),
            1,
            ComplexType(i1),
            ArrayAttr([IntegerAttr(1, 1), IntegerAttr(0, 1)]),
        ),
        (
            (1, 1),
            1,
            ComplexType(i1),
            ArrayAttr([IntegerAttr(1, 1), IntegerAttr(1, 1)]),
        ),
        (
            (0, 1),
            1,
            ComplexType(i1),
            ArrayAttr([BoolAttr(0, 1), BoolAttr(1, 1)]),
        ),
        (
            (0, 1),
            1,
            ComplexType(i1),
            ArrayAttr(
                [
                    BoolAttr(False, i1),
                    BoolAttr(True, i1),
                ]
            ),
        ),
        (
            (1, 2),
            16,
            ComplexType(f16),
            ArrayAttr([FloatAttr(1, 16), FloatAttr(2, 16)]),
        ),
        (
            (1.2, 4.0),
            32,
            ComplexType(f32),
            ArrayAttr([FloatAttr(1.2, 32), FloatAttr(4.0, 32)]),
        ),
        (
            (1, 4),
            64,
            ComplexType(f64),
            ArrayAttr([FloatAttr(1, 64), FloatAttr(4, 64)]),
        ),
        (
            (1, 4),
            80,
            ComplexType(f80),
            ArrayAttr([FloatAttr(1, 80), FloatAttr(4, 80)]),
        ),
        (
            (1, 4),
            128,
            ComplexType(f128),
            ArrayAttr([FloatAttr(1, 128), FloatAttr(4, 128)]),
        ),
    ],
)
def test_complex_constant_construct(
    tuple: tuple[int | float, int | float],
    width: int,
    result_type: ComplexType,
    value_type: ArrayAttr[FloatAttr] | ArrayAttr[BoolAttr],
):
    op = complex.ConstantOp.from_tuple_and_width(tuple, width)
    op2 = complex.ConstantOp(value_type, result_type)

    assert op.result_types[0] == result_type
    assert op.value == value_type
    assert op.result_types[0] == op2.result_types[0]
    assert op.value == op2.value


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (
            ArrayAttr([FloatAttr(2, 16), FloatAttr(3.3, 32)]),
            ComplexType(f16),
        ),
        (
            ArrayAttr([FloatAttr(2, 32), FloatAttr(3.3, 32)]),
            ComplexType(f64),
        ),
        (
            ArrayAttr([BoolAttr(0, 1), FloatAttr(3.3, 32)]),
            ComplexType(f64),
        ),
        (
            ArrayAttr([FloatAttr(2, 32), BoolAttr(1, 1)]),
            ComplexType(i1),
        ),
        (
            ArrayAttr([BoolAttr(0, 1), BoolAttr(1, 1)]),
            ComplexType(f16),
        ),
    ],
)
def test_complex_constant_construct_incorrect(
    input: ArrayAttr, expected_output: ComplexType
):
    with pytest.raises(VerifyException):
        op = complex.ConstantOp(input, expected_output)
        op.verify_()


class Test_complex_unary_float_result_construction:
    a = complex.ConstantOp.from_tuple_and_width((2.23, 45.5), 16)

    @pytest.mark.parametrize(
        "func",
        [
            complex.AbsOp,
            complex.AngleOp,
            complex.ReOp,
            complex.ImOp,
        ],
    )
    @pytest.mark.parametrize(
        "flags",
        [
            FastMathFlagsAttr("none"),
            FastMathFlagsAttr("fast"),
            None,
        ],
    )
    def test_complex_unary_ops_init(
        self,
        func: type[complex.ComplexUnaryRealResultOperation],
        flags: FastMathFlagsAttr | None,
    ):
        op = func(self.a, flags)
        assert op.result.type == f16
        assert op.complex.type == ComplexType(f16)
        assert op.complex.type == self.a.complex.type
        assert op.complex.owner is self.a
        assert op.result.type == self.a.complex.type.element_type
        assert op.fastmath == (flags or FastMathFlagsAttr("none"))


class Test_complex_unary_construction:
    a = complex.ConstantOp.from_tuple_and_width((2.23, 45.5), 16)

    @pytest.mark.parametrize(
        "func",
        [
            complex.ConjOp,
            complex.CosOp,
            complex.ExpOp,
            complex.Expm1Op,
            complex.LogOp,
            complex.Log1pOp,
            complex.NegOp,
            complex.RsqrtOp,
            complex.SignOp,
            complex.SinOp,
            complex.SqrtOp,
            complex.TanOp,
            complex.TanhOp,
        ],
    )
    @pytest.mark.parametrize(
        "flags",
        [
            FastMathFlagsAttr("none"),
            FastMathFlagsAttr("fast"),
            None,
        ],
    )
    def test_complex_unary_ops_init(
        self,
        func: type[complex.ComplexUnaryComplexResultOperation],
        flags: FastMathFlagsAttr | None,
    ):
        op = func(self.a, flags)
        assert op.result.type == ComplexType(f16)
        assert op.result.type == op.complex.type
        assert op.complex.type == self.a.complex.type
        assert op.complex.owner is self.a
        assert op.result.type == self.a.complex.type
        assert op.fastmath == (flags or FastMathFlagsAttr("none"))


class Test_complex_binary_construction:
    a = complex.ConstantOp.from_tuple_and_width((2, 4), 32)
    b = complex.ConstantOp.from_tuple_and_width((5, 7), 32)

    def test_constant_construct(self):
        assert self.a.result_types[0] == ComplexType(f32)
        assert self.a.value.data[0] == FloatAttr(2, f32)
        assert self.a.value.data[1] == FloatAttr(4, f32)

    @pytest.mark.parametrize(
        "func",
        [
            complex.AddOp,
            complex.Atan2Op,
            complex.DivOp,
            complex.MulOp,
            complex.PowOp,
            complex.SubOp,
        ],
    )
    @pytest.mark.parametrize(
        "flags",
        [
            FastMathFlagsAttr("none"),
            FastMathFlagsAttr("fast"),
            None,
        ],
    )
    def test_complex_ops(
        self,
        func: type[complex.ComplexBinaryOp],
        flags: FastMathFlagsAttr | None,
    ):
        op = func(self.a, self.b, flags)
        assert op.operands[0].owner is self.a
        assert op.operands[1].owner is self.b
        assert op.fastmath == (flags or FastMathFlagsAttr("none"))


@pytest.mark.parametrize(
    "lhs_type, rhs_type, is_correct",
    [
        (f32, f32, True),
        (f16, f16, True),
        (f64, f64, True),
        (f128, f128, True),
        (f16, f64, False),
        (i16, f64, False),
    ],
)
def test_create_op(lhs_type: Attribute, rhs_type: Attribute, is_correct: bool):
    lhs = create_ssa_value(lhs_type)
    rhs = create_ssa_value(rhs_type)

    if is_correct:
        op = complex.CreateOp(lhs, rhs, ComplexType(lhs_type))
        op.verify_()
        assert op.real == lhs
        assert op.imaginary == rhs
        assert op.operand_types[0] == lhs_type
        assert op.operand_types[1] == lhs_type
        assert op.result_types[0] == ComplexType(lhs_type)
    else:
        with pytest.raises(VerifyException):
            op = complex.CreateOp(lhs, rhs, ComplexType(lhs_type))
            op.verify_()


@pytest.mark.parametrize(
    "func",
    [
        complex.EqualOp,
        complex.NotEqualOp,
    ],
)
def test_complex_compare_ops(func: type[complex.ComplexCompareOp]):
    lhs = complex.ConstantOp.from_tuple_and_width((3.3, 56), 32)
    rhs = complex.ConstantOp.from_tuple_and_width((2.2, 4.5), 32)

    op = func(lhs, rhs)
    op.verify_()
    assert op.lhs == lhs.complex
    assert op.rhs == rhs.complex
    assert op.result_types[0] == i1


@pytest.mark.parametrize(
    "func",
    [
        complex.EqualOp,
        complex.NotEqualOp,
    ],
)
def test_complex_compare_ops_incorrect(func: type[complex.ComplexCompareOp]):
    lhs = complex.ConstantOp.from_tuple_and_width((3.3, 56), 32)
    rhs = complex.ConstantOp.from_tuple_and_width((2.2, 4.5), 16)

    op = func(lhs, rhs)
    with pytest.raises(VerifyException):
        op.verify_()
