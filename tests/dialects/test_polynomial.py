import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    f64,
)
from xdsl.dialects.polynomial import (
    ChebyshevPolynomialAttr,
    PolynomialType,
    RingAttr,
    TypedChebyshevPolynomialAttr,
)

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
    assert typed.coeff_values == [1.0, 2.0, 3.0]
    assert typed.type == poly_ty


def test_typed_chebyshev_polynomial_from_attr():
    poly_ty = PolynomialType(RingAttr(f64))
    inner = ChebyshevPolynomialAttr((1.0, 2.0))
    typed = TypedChebyshevPolynomialAttr(poly_ty, inner)
    assert typed.value is inner
