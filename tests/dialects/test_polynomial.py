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
    EvalOp,
)
from xdsl.ir import Attribute
from xdsl.utils.test_value import create_ssa_value

# --- ChebyshevPolynomialAttr construction ---


@pytest.mark.parametrize(
    "coefficients, expected_degree, expected_coeffs",
    [
        pytest.param((0.5, 1.2, 0.3), 2, [0.5, 1.2, 0.3], id="float_tuple"),
        pytest.param(
            ArrayAttr([FloatAttr(1.0, f64), FloatAttr(2.0, f64)]),
            1,
            [1.0, 2.0],
            id="array_attr",
        ),
        pytest.param((42.0,), 0, [42.0], id="single_coefficient"),
    ],
)
def test_attr_construction(
    coefficients: tuple[float, ...] | ArrayAttr[FloatAttr],
    expected_degree: int,
    expected_coeffs: list[float],
):
    attr = ChebyshevPolynomialAttr(coefficients)
    assert attr.degree == expected_degree
    assert attr.coeff_values == expected_coeffs


# --- EvalOp construction ---


def test_eval_basic_construction():
    x = create_ssa_value(f32)
    op = EvalOp(x, (0.5, 1.2, 0.3))
    assert op.value.type == f32
    assert op.result.type == f32
    assert op.degree == 2
    assert op.polynomial.coeff_values == [0.5, 1.2, 0.3]


def test_eval_pre_built_attr():
    attr = ChebyshevPolynomialAttr((1.0, 2.0, 3.0))
    x = create_ssa_value(f64)
    op = EvalOp(x, attr)
    assert op.polynomial is attr


@pytest.mark.parametrize(
    "tp",
    [f16, f32, f64, VectorType(f32, [4]), TensorType(f64, [8])],
)
def test_eval_type_polymorphism(tp: Attribute):
    x = create_ssa_value(tp)
    op = EvalOp(x, (1.0, 2.0))
    assert op.value.type == tp
    assert op.result.type == tp


def test_eval_domain_bounds_propagated():
    x = create_ssa_value(f32)
    op = EvalOp(x, (1.0, 2.0), domain_lower=-10.0, domain_upper=0.0)
    assert op.domain_lower is not None
    assert op.domain_upper is not None
    assert op.domain_lower.value.data == -10.0
    assert op.domain_upper.value.data == 0.0
