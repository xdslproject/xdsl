import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
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
    Polynomial,
    PolynomialType,
    RingAttr,
    TypedChebyshevPolynomialAttr,
)
from xdsl.ir import Attribute
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError
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


# --- EvalOp construction ---


def test_eval_basic_construction():
    x = create_ssa_value(f32)
    op = EvalOp(x, (0.5, 1.2, 0.3), EvalScheme.CLENSHAW)
    assert op.value.type == f32
    assert op.result.type == f32
    assert op.degree == 2
    assert op.polynomial.coeff_values == [0.5, 1.2, 0.3]
    assert op.scheme.data == EvalScheme.CLENSHAW.value
    assert op.eval_scheme == EvalScheme.CLENSHAW


def test_eval_pre_built_typed_attr():
    poly_ty = PolynomialType(RingAttr(f64))
    typed = TypedChebyshevPolynomialAttr(poly_ty, (1.0, 2.0, 3.0))
    x = create_ssa_value(f64)
    op = EvalOp(x, typed, EvalScheme.CLENSHAW)
    assert op.polynomial is typed


def test_eval_pre_built_untyped_attr_is_wrapped():
    """An untyped ChebyshevPolynomialAttr is auto-wrapped with the default type."""
    untyped = ChebyshevPolynomialAttr((1.0, 2.0, 3.0))
    x = create_ssa_value(f64)
    op = EvalOp(x, untyped, EvalScheme.CLENSHAW)
    assert isinstance(op.polynomial, TypedChebyshevPolynomialAttr)
    assert op.polynomial.value is untyped


@pytest.mark.parametrize(
    "tp",
    [f16, f32, f64, VectorType(f32, [4]), TensorType(f64, [8])],
)
def test_eval_type_polymorphism(tp: Attribute):
    x = create_ssa_value(tp)
    op = EvalOp(x, (1.0, 2.0), EvalScheme.CLENSHAW)
    assert op.value.type == tp
    assert op.result.type == tp


def test_eval_domain_bounds_propagated():
    x = create_ssa_value(f32)
    op = EvalOp(
        x, (1.0, 2.0), EvalScheme.CLENSHAW, domain_lower=-10.0, domain_upper=0.0
    )
    assert op.domain_lower is not None
    assert op.domain_upper is not None
    assert op.domain_lower.value.data == -10.0
    assert op.domain_upper.value.data == 0.0


def test_eval_scheme_propagated():
    x = create_ssa_value(f32)
    op = EvalOp(x, (1.0, 2.0), EvalScheme.CLENSHAW)
    assert op.scheme.data == EvalScheme.CLENSHAW.value
    assert op.eval_scheme == EvalScheme.CLENSHAW


def test_eval_scheme_accepts_string_attr():
    x = create_ssa_value(f32)
    attr = StringAttr("clenshaw")
    op = EvalOp(x, (1.0, 2.0), attr)
    assert op.scheme is attr
    assert op.eval_scheme == EvalScheme.CLENSHAW


def test_eval_scheme_accepts_str():
    x = create_ssa_value(f32)
    op = EvalOp(x, (1.0, 2.0), "clenshaw")
    assert op.scheme.data == "clenshaw"
    assert op.eval_scheme == EvalScheme.CLENSHAW


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
