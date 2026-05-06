import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, f64
from xdsl.dialects.polynomial import (
    Polynomial,
    PolynomialType,
    RingAttr,
)
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError


def test_ring_attr_construction():
    ring = RingAttr(f64)
    assert ring.coefficient_type == f64


def test_polynomial_type_construction():
    poly_ty = PolynomialType(RingAttr(f64))
    assert poly_ty.ring.coefficient_type == f64


def _parse_type(src: str):
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Polynomial)
    return Parser(ctx, src).parse_type()


def test_parse_wrong_attr_kind():
    # `parse_optional_attribute` returns a non-RingAttr -> error branch
    with pytest.raises(ParseError, match="expected RingAttr"):
        _parse_type("!polynomial.polynomial<ring = 42 : i32>")
