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


Polynomial = Dialect(
    "polynomial",
    [],
    [
        RingAttr,
        PolynomialType,
    ],
)
