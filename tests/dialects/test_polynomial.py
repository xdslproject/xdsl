import pytest

from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    FloatAttr,
    StringAttr,
    TensorType,
    VectorType,
    f16,
    f32,
    f64,
)
from xdsl.dialects.polynomial import (
    ChebyshevPolynomialAttr,
    EvalOp,
    EvalScheme,
    PolynomialType,
    RingAttr,
    TypedChebyshevPolynomialAttr,
)
from xdsl.ir import Attribute
from xdsl.utils.test_value import create_ssa_value

# --- ChebyshevPolynomialAttr construction ---


@pytest.mark.parametrize(
    "coefficients, expected_degree, expected_coeffs",
    [
        pytest.param((0.5, 1.2, 0.3), 2, (0.5, 1.2, 0.3), id="float_tuple"),
        pytest.param(
            ArrayAttr([FloatAttr(1.0, f64), FloatAttr(2.0, f64)]),
            1,
            (1.0, 2.0),
            id="array_attr",
        ),
        pytest.param((42.0,), 0, (42.0,), id="single_coefficient"),
    ],
)
def test_attr_construction(
    coefficients: tuple[float, ...] | ArrayAttr[FloatAttr],
    expected_degree: int,
    expected_coeffs: tuple[float, ...],
):
    attr = ChebyshevPolynomialAttr(coefficients)
    assert attr.degree == expected_degree
    assert attr.coeff_values == expected_coeffs


# --- RingAttr / PolynomialType / TypedChebyshevPolynomialAttr ---


def test_ring_attr_construction():
    ring = RingAttr(f64)
    assert ring.coefficient_type == f64


def test_polynomial_type_construction():
    poly_ty = PolynomialType(RingAttr(f64))
    assert poly_ty.ring.coefficient_type == f64


def test_typed_chebyshev_polynomial_from_tuple():
    poly_ty = PolynomialType(RingAttr(f64))
    typed = TypedChebyshevPolynomialAttr(poly_ty, (1.0, 2.0, 3.0))
    assert typed.degree == 2
    assert typed.coeff_values == (1.0, 2.0, 3.0)
    assert typed.type == poly_ty


def test_typed_chebyshev_polynomial_from_attr():
    poly_ty = PolynomialType(RingAttr(f64))
    inner = ChebyshevPolynomialAttr((1.0, 2.0))
    typed = TypedChebyshevPolynomialAttr(poly_ty, inner)
    assert typed.value is inner


# --- EvalOp construction ---


def test_eval_init_stores_attributes_verbatim():
    """`__init__` takes pre-built attributes and stores them as-is."""
    poly_ty = PolynomialType(RingAttr(f64))
    typed = TypedChebyshevPolynomialAttr(poly_ty, (1.0, 2.0, 3.0))
    scheme_attr = StringAttr("clenshaw")
    x = create_ssa_value(f64)
    op = EvalOp(x, typed, scheme_attr)
    assert op.polynomial is typed
    assert op.scheme is scheme_attr


def test_eval_get_basic_construction():
    x = create_ssa_value(f32)
    op = EvalOp.get(x, (0.5, 1.25, 0.375), f32, EvalScheme.CLENSHAW)
    assert op.value.type == f32
    assert op.result.type == f32
    assert op.degree == 2
    assert op.polynomial.coeff_values == (0.5, 1.25, 0.375)
    assert op.polynomial.type == PolynomialType(RingAttr(f32))
    assert op.scheme.data == EvalScheme.CLENSHAW.value
    assert op.eval_scheme == EvalScheme.CLENSHAW


@pytest.mark.parametrize(
    "value_type, element_type",
    [
        (f16, f16),
        (f32, f32),
        (f64, f64),
        (VectorType(f32, [4]), f32),
        (TensorType(f64, [8]), f64),
    ],
)
def test_eval_get_type_polymorphism(value_type: Attribute, element_type: AnyFloat):
    x = create_ssa_value(value_type)
    op = EvalOp.get(x, (1.0, 2.0), element_type, EvalScheme.CLENSHAW)
    assert op.value.type == value_type
    assert op.result.type == value_type
    assert op.polynomial.type == PolynomialType(RingAttr(element_type))


def test_eval_get_domain_bounds_use_element_type():
    x = create_ssa_value(f32)
    op = EvalOp.get(
        x,
        (1.0, 2.0),
        f32,
        EvalScheme.CLENSHAW,
        domain_lower=-10.0,
        domain_upper=0.0,
    )
    assert op.domain_lower is not None
    assert op.domain_upper is not None
    assert op.domain_lower.value.data == -10.0
    assert op.domain_upper.value.data == 0.0
    assert op.domain_lower.type == f32
    assert op.domain_upper.type == f32


def test_eval_get_accepts_str_scheme():
    x = create_ssa_value(f32)
    op = EvalOp.get(x, (1.0, 2.0), f32, "clenshaw")
    assert op.scheme.data == "clenshaw"
    assert op.eval_scheme == EvalScheme.CLENSHAW
