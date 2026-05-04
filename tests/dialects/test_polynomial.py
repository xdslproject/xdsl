import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
    FloatAttr,
    f64,
)
from xdsl.dialects.polynomial import (
    ChebyshevPolynomialAttr,
    Polynomial,
    PolynomialType,
    RingAttr,
    TypedChebyshevPolynomialAttr,
)
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError

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


# --- Parsing ---


def _parse_type(src: str):
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Polynomial)
    return Parser(ctx, src).parse_type()


def test_parse_wrong_attr_kind():
    # `parse_optional_attribute` returns a non-RingAttr -> error branch
    with pytest.raises(ParseError, match="expected RingAttr"):
        _parse_type("!polynomial.polynomial<ring = 42 : i32>")
