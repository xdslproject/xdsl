from xdsl.dialects.builtin import f64
from xdsl.dialects.polynomial import (
    PolynomialType,
    RingAttr,
)


def test_ring_attr_construction():
    ring = RingAttr(f64)
    assert ring.coefficient_type == f64


def test_polynomial_type_construction():
    poly_ty = PolynomialType(RingAttr(f64))
    assert poly_ty.ring.coefficient_type == f64
