"""
A minimal polynomial dialect for representing unevaluated polynomial
approximations, designed for use with equality saturation.

Cost estimation can be done on a single `polynomial.eval_chebyshev` op
that carries all the information needed. After extraction, the selected
polynomial variant will be expanded into arithmetic operations.

https://heir.dev/docs/dialects/polynomial/

"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import (
    AnyFloatConstr,
    ArrayAttr,
    ContainerOf,
    FloatAttr,
    f64,
)
from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure, SameOperandsAndResultType
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class ChebyshevPolynomialAttr(ParametrizedAttribute):
    """
    Chebyshev polynomial approximation on a specific domain.


    Syntax: #polynomial.chebyshev<[coefficients], domain_lower, domain_upper>
    Example:
        #polynomial.chebyshev<[0.5 : f64, 1.2 : f64, 0.3 : f64],
                               -1.0 : f64, 1.0 : f64>
    """

    name = "polynomial.chebyshev"

    coefficients: ArrayAttr[FloatAttr]
    domain_lower: FloatAttr
    domain_upper: FloatAttr

    def __init__(
        self,
        coefficients: tuple[float, ...] | ArrayAttr[FloatAttr],
        domain_lower: float | FloatAttr = FloatAttr(-1.0, f64),
        domain_upper: float | FloatAttr = FloatAttr(1.0, f64),
    ):
        if isinstance(coefficients, ArrayAttr):
            arr = coefficients
        else:
            arr = ArrayAttr([FloatAttr(c, f64) for c in coefficients])
        if not isinstance(domain_lower, FloatAttr):
            domain_lower = FloatAttr(float(domain_lower), f64)
        if not isinstance(domain_upper, FloatAttr):
            domain_upper = FloatAttr(float(domain_upper), f64)
        super().__init__(arr, domain_lower, domain_upper)

    @property
    def degree(self) -> int:
        """Polynomial degree (number of coefficients minus one)."""
        return len(self.coefficients) - 1

    @property
    def coeff_values(self) -> list[float]:
        """Extract coefficient values as Python floats."""
        return [c.value.data for c in self.coefficients]

    @property
    def lower(self) -> float:
        """Lower bound of the approximation domain."""
        return self.domain_lower.value.data

    @property
    def upper(self) -> float:
        """Upper bound of the approximation domain."""
        return self.domain_upper.value.data


@irdl_op_definition
class EvalChebyshevOp(IRDLOperation):
    """
    Evaluate a Chebyshev polynomial at a given point.

    This op is *unevaluated* -- it carries all information needed for:
      - Cost estimation (degree, accuracy from coefficients)
      - Later lowering to arithmetic ops (Clenshaw's algorithm)

    The polynomial attribute contains the coefficients and domain bounds.
    During lowering, the input value is mapped from [lower, upper] to
    [-1, 1] before Clenshaw evaluation.

    Example:
        %result = polynomial.eval_chebyshev %x
            #polynomial.chebyshev<[0.5 : f64, 1.2 : f64],
                                   -1.0 : f64, 1.0 : f64>
            : f32
    """

    name = "polynomial.eval_chebyshev"

    T: ClassVar = VarConstraint("T", ContainerOf(AnyFloatConstr))

    value = operand_def(T)
    result = result_def(T)

    polynomial = prop_def(ChebyshevPolynomialAttr)

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = "$value $polynomial attr-dict `:` type($result)"

    def __init__(
        self,
        value: Operation | SSAValue,
        polynomial: ChebyshevPolynomialAttr | tuple[float, ...],
        domain_lower: float | FloatAttr = FloatAttr(-1.0, f64),
        domain_upper: float | FloatAttr = FloatAttr(1.0, f64),
    ):
        if not isinstance(polynomial, ChebyshevPolynomialAttr):
            polynomial = ChebyshevPolynomialAttr(polynomial, domain_lower, domain_upper)

        value = SSAValue.get(value)

        super().__init__(
            operands=[value],
            result_types=[value.type],
            properties={
                "polynomial": polynomial,
            },
        )

    def verify_(self) -> None:
        if self.polynomial.lower >= self.polynomial.upper:
            raise VerifyException(
                f"domain_lower ({self.polynomial.lower}) must be strictly "
                f"less than domain_upper ({self.polynomial.upper})"
            )
        if self.polynomial.degree < 1:
            raise VerifyException(
                "Chebyshev polynomial must have at least degree 1 "
                f"(got {self.polynomial.degree + 1} coefficients)"
            )

    @property
    def degree(self) -> int:
        return self.polynomial.degree


Polynomial = Dialect(
    "polynomial",
    [
        EvalChebyshevOp,
    ],
    [
        ChebyshevPolynomialAttr,
    ],
)
