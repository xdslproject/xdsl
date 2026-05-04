"""
A polynomial dialect for representing unevaluated polynomial
approximations, designed for use with equality saturation.

Cost estimation can be done on a single `polynomial.eval` op
that carries all the information needed. After extraction, the selected
polynomial variant will be expanded into arithmetic operations.

[HEIR Documentation](https://heir.dev/docs/dialects/polynomial/)

"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    f64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class RingAttr(ParametrizedAttribute):
    """
    A polynomial ring, parameterized by the coefficient type.

    Syntax: #polynomial.ring<coefficientType=f64>
    """

    name = "polynomial.ring"

    coefficient_type: Attribute

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_keyword("coefficientType")
            parser.parse_punctuation("=")
            coeff_type = parser.parse_type()
        return (coeff_type,)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<coefficientType = ")
        printer.print_attribute(self.coefficient_type)
        printer.print_string(">")


@irdl_attr_definition
class PolynomialType(ParametrizedAttribute, TypeAttribute):
    """
    Type of an element of a polynomial ring.

    Syntax: !polynomial.polynomial<ring=<coefficientType=f64>>
    """

    name = "polynomial.polynomial"

    ring: RingAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_keyword("ring")
            parser.parse_punctuation("=")
            # Accept either inline form `<coefficientType=...>` or
            # attribute reference like `#alias` / `#polynomial.ring<...>`.
            # HEIR's printer often hoists ring attributes into top-level
            # aliases, so both need to be handled on round-trip.
            ring = parser.parse_optional_attribute()
            if ring is None:
                ring_params = RingAttr.parse_parameters(parser)
                ring = RingAttr.new(ring_params)
            elif not isinstance(ring, RingAttr):
                parser.raise_error(f"expected RingAttr in polynomial type, got {ring}")
        return (ring,)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<ring = ")
        self.ring.print_parameters(printer)
        printer.print_string(">")


@irdl_attr_definition
class ChebyshevPolynomialAttr(ParametrizedAttribute):
    """
    Untyped Chebyshev polynomial with double precision floating point coefficients.

    For use directly inside `polynomial.eval`, prefer `TypedChebyshevPolynomialAttr`,
    which carries an explicit polynomial type and matches HEIR's `polynomial.eval`
    op signature.

    Syntax: #polynomial.chebyshev_polynomial<[coefficients]>
    Example:
        #polynomial.chebyshev_polynomial<[0.5 : f64, 1.2 : f64, 0.3 : f64]>
    """

    name = "polynomial.chebyshev_polynomial"

    coefficients: ArrayAttr[FloatAttr]

    def __init__(
        self,
        coefficients: tuple[float, ...] | ArrayAttr[FloatAttr],
    ):
        if isinstance(coefficients, ArrayAttr):
            arr = coefficients
        else:
            arr = ArrayAttr([FloatAttr(c, f64) for c in coefficients])
        super().__init__(arr)

    @property
    def degree(self) -> int:
        """Polynomial degree (number of coefficients minus one)."""
        return len(self.coefficients) - 1

    @property
    def coeff_values(self) -> list[float]:
        """Extract coefficient values as Python floats."""
        return [c.value.data for c in self.coefficients]


@irdl_attr_definition
class TypedChebyshevPolynomialAttr(ParametrizedAttribute):
    """
    Chebyshev polynomial with an explicit polynomial type.

    Syntax: #polynomial.typed_chebyshev_polynomial<[coefficients]> : !polynomial.polynomial<...>
    Example:
        #polynomial.typed_chebyshev_polynomial<[1.0, 2.0]> :
            !polynomial.polynomial<ring=<coefficientType=f64>>
    """

    name = "polynomial.typed_chebyshev_polynomial"

    type: Attribute
    value: ChebyshevPolynomialAttr

    def __init__(
        self,
        type: Attribute,
        value: ChebyshevPolynomialAttr | tuple[float, ...],
    ):
        if not isinstance(value, ChebyshevPolynomialAttr):
            value = ChebyshevPolynomialAttr(value)
        super().__init__(type, value)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        # Accept either inline form or full attribute
        # references (`#alias`, `#polynomial.typed_chebyshev_polynomial<...>`).
        # HEIR's printer often hoists ring attributes into top-level
        # aliases, so both need to be handled on round-trip.
        attr = parser.parse_optional_attribute()
        if attr is not None:
            if not isinstance(attr, TypedChebyshevPolynomialAttr):
                parser.raise_error(f"expected TypedChebyshevPolynomialAttr, got {attr}")
            return (attr.type, attr.value)
        with parser.in_angle_brackets():
            coeffs = parser.parse_attribute()
        parser.parse_punctuation(":")
        poly_type = parser.parse_type()
        value = ChebyshevPolynomialAttr.new((coeffs,))
        return (poly_type, value)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.value.coefficients)
        printer.print_string("> : ")
        printer.print_attribute(self.type)

    @property
    def degree(self) -> int:
        return self.value.degree

    @property
    def coeff_values(self) -> list[float]:
        return self.value.coeff_values


Polynomial = Dialect(
    "polynomial",
    [],
    [
        ChebyshevPolynomialAttr,
        TypedChebyshevPolynomialAttr,
        RingAttr,
        PolynomialType,
    ],
)
