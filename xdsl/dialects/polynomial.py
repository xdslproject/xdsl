"""
A polynomial dialect for representing unevaluated polynomial
approximations, designed for use with equality saturation.

Cost estimation can be done on a single `polynomial.eval` op
that carries all the information needed. After extraction, the selected
polynomial variant will be expanded into arithmetic operations.

[HEIR Documentation](https://heir.dev/docs/dialects/polynomial/)

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
    opt_attr_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure, SameOperandsAndResultType
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class ChebyshevPolynomialAttr(ParametrizedAttribute):
    """
    Chebyshev polynomial with double precision floating point coefficients.

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


@irdl_op_definition
class EvalOp(IRDLOperation):
    """
    Evaluate a polynomial at a given point.

    This op is *unevaluated* -- it carries all information needed for:
      - Cost estimation (degree, accuracy from coefficients)
      - Later lowering to arithmetic ops (Clenshaw's algorithm)

    The optional domain_lower/domain_upper attributes specify the input
    domain for Clenshaw evaluation. During lowering, the input value is
    mapped from [lower, upper] to [-1, 1] before evaluation.

    Example:
        %result = polynomial.eval
            #polynomial.chebyshev_polynomial<[0.5 : f64, 1.2 : f64]>,
            %x {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}
            : f32
    """

    name = "polynomial.eval"

    T: ClassVar = VarConstraint("T", ContainerOf(AnyFloatConstr))

    value = operand_def(T)
    result = result_def(T)

    polynomial = prop_def(ChebyshevPolynomialAttr)

    domain_lower = opt_attr_def(FloatAttr)
    domain_upper = opt_attr_def(FloatAttr)

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = "$polynomial `,` $value attr-dict `:` type($value)"

    def __init__(
        self,
        value: Operation | SSAValue,
        polynomial: ChebyshevPolynomialAttr | tuple[float, ...],
        domain_lower: float | FloatAttr | None = None,
        domain_upper: float | FloatAttr | None = None,
    ):
        if not isinstance(polynomial, ChebyshevPolynomialAttr):
            polynomial = ChebyshevPolynomialAttr(polynomial)

        value = SSAValue.get(value)

        if isinstance(domain_lower, (int, float)):
            domain_lower = FloatAttr(float(domain_lower), f64)
        if isinstance(domain_upper, (int, float)):
            domain_upper = FloatAttr(float(domain_upper), f64)

        attrs: dict[str, FloatAttr] = {}
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

    @property
    def degree(self) -> int:
        return self.polynomial.degree


Polynomial = Dialect(
    "polynomial",
    [
        EvalOp,
    ],
    [
        ChebyshevPolynomialAttr,
    ],
)
