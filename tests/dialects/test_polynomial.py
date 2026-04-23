import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    TensorType,
    VectorType,
    f16,
    f32,
    f64,
)
from xdsl.dialects.polynomial import (
    ChebyshevPolynomialAttr,
    EvalChebyshevOp,
)
from xdsl.ir import Attribute
from xdsl.utils.test_value import create_ssa_value

# --- ChebyshevPolynomialAttr construction ---


@pytest.mark.parametrize(
    "coefficients, expected_degree, expected_coeffs",
    [
        ((0.5, 1.2, 0.3), 2, [0.5, 1.2, 0.3]),
        (ArrayAttr([FloatAttr(1.0, f64), FloatAttr(2.0, f64)]), 1, [1.0, 2.0]),
        ((42.0,), 0, [42.0]),
    ],
    ids=["float_tuple", "array_attr", "single_coefficient"],
)
def test_attr_default_domain(
    coefficients: tuple[float, ...] | ArrayAttr[FloatAttr],
    expected_degree: int,
    expected_coeffs: list[float],
):
    attr = ChebyshevPolynomialAttr(coefficients)
    assert attr.degree == expected_degree
    assert attr.coeff_values == expected_coeffs
    assert attr.lower == -1.0
    assert attr.upper == 1.0


@pytest.mark.parametrize(
    "coefficients, lower, upper, expected_degree, expected_coeffs",
    [
        ((0.5, 1.2), -2.0, 3.0, 1, [0.5, 1.2]),
        ((1.0, 2.0), -10.0, 0.0, 1, [1.0, 2.0]),
    ],
    ids=["short_domain", "negative_domain"],
)
def test_attr_custom_domain(
    coefficients: tuple[float, ...],
    lower: float,
    upper: float,
    expected_degree: int,
    expected_coeffs: list[float],
):
    attr = ChebyshevPolynomialAttr(coefficients, domain_lower=lower, domain_upper=upper)
    assert attr.degree == expected_degree
    assert attr.coeff_values == expected_coeffs
    assert attr.lower == lower
    assert attr.upper == upper


# --- EvalChebyshevOp construction ---


def test_eval_basic_construction():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, (0.5, 1.2, 0.3))
    assert op.value.type == f32
    assert op.result.type == f32
    assert op.degree == 2
    assert op.polynomial.coeff_values == [0.5, 1.2, 0.3]


def test_eval_pre_built_attr():
    attr = ChebyshevPolynomialAttr((1.0, 2.0, 3.0), -5.0, 5.0)
    x = create_ssa_value(f64)
    op = EvalChebyshevOp(x, attr)
    assert op.polynomial is attr
    assert op.polynomial.lower == -5.0
    assert op.polynomial.upper == 5.0


@pytest.mark.parametrize(
    "tp",
    [f16, f32, f64, VectorType(f32, [4]), TensorType(f64, [8])],
    ids=["f16", "f32", "f64", "vector<4xf32>", "tensor<8xf64>"],
)
def test_eval_type_polymorphism(tp: Attribute):
    x = create_ssa_value(tp)
    op = EvalChebyshevOp(x, (1.0, 2.0))
    assert op.value.type == tp
    assert op.result.type == tp


def test_eval_domain_bounds_propagated():
    x = create_ssa_value(f32)
    op = EvalChebyshevOp(x, (1.0, 2.0), domain_lower=-10.0, domain_upper=0.0)
    assert op.polynomial.lower == -10.0
    assert op.polynomial.upper == 0.0
