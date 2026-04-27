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
from typing import ClassVar

from xdsl.dialects.builtin import (
    AnyFloatConstr,
    ArrayAttr,
    ContainerOf,
    FloatAttr,
    StringAttr,
    f64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Pure, SameOperandsAndResultType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class EvalScheme(StrEnum):
    """
    Evaluation scheme used to lower a `polynomial.eval` op to arithmetic ops.

    Stored on the op as a builtin string attribute so that compilers like HEIR
    that don't know about this xDSL extension can preserve it on round-trip
    (the same way they ignore `domain_lower`/`domain_upper`).
    """

    CLENSHAW = "clenshaw"


@irdl_attr_definition
class RingAttr(ParametrizedAttribute):
    """
    A polynomial ring, parameterized by the coefficient type.

    Syntax: #polynomial.ring<coefficientType=f64>
    """

    name = "polynomial.ring"

    coefficient_type: Attribute

    def __init__(self, coefficient_type: Attribute):
        super().__init__(coefficient_type)

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

    def __init__(self, ring: RingAttr):
        super().__init__(ring)

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


def _default_polynomial_type() -> PolynomialType:
    """Default polynomial type for f64 Chebyshev coefficients."""
    return PolynomialType(RingAttr(f64))


@irdl_op_definition
class EvalOp(IRDLOperation):
    """
    Evaluate a polynomial at a given point.

    This op is *unevaluated* but carries all information needed for
    later lowering to arithmetic ops, dispatched on `scheme`.

    Syntax: polynomial.eval $polynomial `,` $value attr-dict `:` type($value)
    Example:
        %result = polynomial.eval
            #polynomial.typed_chebyshev_polynomial<[0.5, 1.2]> :
                !polynomial.polynomial<ring=<coefficientType=f64>>,
            %x {scheme = "clenshaw",
                domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}
            : f32
    """

    name = "polynomial.eval"

    T: ClassVar = VarConstraint("T", ContainerOf(AnyFloatConstr))

    value = operand_def(T)
    result = result_def(T)

    polynomial = prop_def(TypedChebyshevPolynomialAttr)

    scheme = attr_def(StringAttr)
    domain_lower = opt_attr_def(FloatAttr)
    domain_upper = opt_attr_def(FloatAttr)

    traits = traits_def(Pure(), SameOperandsAndResultType())

    # `qualified(...)` forces the full `#polynomial.typed_chebyshev_polynomial<...>`
    # form when printing/parsing, instead of the elided `<...>` form. This is
    # required for HEIR round-trip: HEIR's parser only accepts the qualified form.
    assembly_format = "qualified($polynomial) `,` $value attr-dict `:` type($value)"

    def __init__(
        self,
        value: Operation | SSAValue,
        polynomial: (
            TypedChebyshevPolynomialAttr | ChebyshevPolynomialAttr | tuple[float, ...]
        ),
        scheme: StringAttr | EvalScheme | str,
        domain_lower: float | FloatAttr | None = None,
        domain_upper: float | FloatAttr | None = None,
    ):
        if isinstance(polynomial, tuple):
            polynomial = TypedChebyshevPolynomialAttr(
                _default_polynomial_type(), polynomial
            )
        elif isinstance(polynomial, ChebyshevPolynomialAttr):
            polynomial = TypedChebyshevPolynomialAttr(
                _default_polynomial_type(), polynomial
            )

        if isinstance(scheme, EvalScheme):
            scheme = StringAttr(scheme.value)
        elif isinstance(scheme, str):
            scheme = StringAttr(scheme)

        value = SSAValue.get(value)

        if isinstance(domain_lower, (int, float)):
            domain_lower = FloatAttr(float(domain_lower), f64)
        if isinstance(domain_upper, (int, float)):
            domain_upper = FloatAttr(float(domain_upper), f64)

        attrs: dict[str, StringAttr | FloatAttr] = {"scheme": scheme}
        if domain_lower is not None:
            attrs["domain_lower"] = domain_lower
        if domain_upper is not None:
            attrs["domain_upper"] = domain_upper

        super().__init__(
            operands=[value],
            result_types=[value.type],
            properties={
                "polynomial": polynomial,
            },
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.domain_lower is not None and self.domain_upper is not None:
            lower = self.domain_lower.value.data
            upper = self.domain_upper.value.data
            if lower >= upper:
                raise VerifyException(
                    f"domain_lower ({lower}) must be strictly "
                    f"less than domain_upper ({upper})"
                )
        if self.polynomial.degree < 1:
            raise VerifyException(
                "Chebyshev polynomial must have at least degree 1 "
                f"(got {self.polynomial.degree + 1} coefficients)"
            )
        try:
            EvalScheme(self.scheme.data)
        except ValueError:
            valid = ", ".join(repr(s.value) for s in EvalScheme)
            raise VerifyException(
                f"unknown evaluation scheme {self.scheme.data!r}; "
                f"expected one of: {valid}"
            )

    @property
    def degree(self) -> int:
        return self.polynomial.degree

    @property
    def eval_scheme(self) -> EvalScheme:
        """Return the scheme as an EvalScheme enum."""
        return EvalScheme(self.scheme.data)


Polynomial = Dialect(
    "polynomial",
    [
        EvalOp,
    ],
    [
        ChebyshevPolynomialAttr,
        TypedChebyshevPolynomialAttr,
        RingAttr,
        PolynomialType,
    ],
)
