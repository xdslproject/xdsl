import math as pymath
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import math, polynomial
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    ModuleOp,
    TensorType,
    VectorType,
)
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

UNDERFLOW_LOWER_BOUND: dict[type, float] = {
    Float16Type: pymath.log(2.0**-24),  # ≈ -16.64
    BFloat16Type: pymath.log(2.0**-133),  # ≈ -92.18
    Float32Type: pymath.log(2.0**-149),  # ≈ -103.28
    Float64Type: pymath.log(2.0**-1074),  # ≈ -744.44
}

# Largest x such that exp(x) is still representable in the given precision.
OVERFLOW_UPPER_BOUND: dict[type, float] = {
    Float16Type: 16.0 * pymath.log(2.0),  # ≈ 11.09
    BFloat16Type: 128.0 * pymath.log(2.0),  # ≈ 88.72
    Float32Type: 128.0 * pymath.log(2.0),  # ≈ 88.72
    Float64Type: 1024.0 * pymath.log(2.0),  # ≈ 709.78
}

# Mantissa width including the hidden bit, per IEEE-754 / brain-float layout.
_PRECISION_BITS: dict[type, int] = {
    Float16Type: 11,  # 10 stored + 1 hidden
    BFloat16Type: 8,  # 7 stored + 1 hidden
    Float32Type: 24,  # 23 stored + 1 hidden
    Float64Type: 53,  # 52 stored + 1 hidden
}

# Hard cap on the polynomial degree the chooser will grow to.
_MAX_DEGREE = 30


def _chebyshev_coefficients(
    f: Callable[[float], float],
    degree: int,
    lower: float,
    upper: float,
) -> list[float]:
    """Chebyshev coefficients c_0..c_n for `f` on [lower, upper] via DCT-I."""
    n = degree
    nodes = [pymath.cos(pymath.pi * j / n) for j in range(n + 1)]
    mid = (upper + lower) / 2.0
    half = (upper - lower) / 2.0
    values = [f(half * t + mid) for t in nodes]
    coeffs: list[float] = []
    for k in range(n + 1):
        s = 0.0
        for j in range(n + 1):
            w = 0.5 if (j == 0 or j == n) else 1.0
            s += w * values[j] * pymath.cos(pymath.pi * k * j / n)
        coeffs.append(2.0 * s / n)
    return coeffs


def _element_float_type(tp: Attribute) -> Attribute:
    if isinstance(tp, (VectorType, TensorType)):
        return cast(Attribute, tp.get_element_type())
    return tp


def _choose_polynomial(
    max_bits_lost: float,
    lower: float,
    upper: float,
    precision_bits: int,
) -> list[float]:
    """Smallest degree such that the Chebyshev-Lobatto *bits-lost* upper bound
    for exp on [lower, upper] in the target precision is <= max_bits_lost.

    bits_lost = log2(ULP error) — the number of low-order mantissa bits that
    are unreliable.
    max_bits_lost = 0 targets 1-ULP-quality (libm-grade); each
    additional unit doubles the allowed ULP error.
    max_bits_lost = -1 targets orrectly-rounded (<= 0.5 ULP).

    Derivation: the conservative Chebyshev approximation bound gives
        |err|_inf <= exp(upper) * width^(d+1) / (2 * 4^d * (d+1)!).
    Dividing by min|exp| = exp(lower) yields the relative error bound:
        rel_err <= exp(width) * width^(d+1) / (2 * 4^d * (d+1)!).
    Multiplying by 2^(precision_bits - 1) converts to ULP error, and taking
    log2 gives bits_lost. Inherits the original bound's ~2x conservatism;
    expect the picked degree to be ~1 higher than empirically optimal.
    """
    width = upper - lower
    degree = 0
    # bits_lost upper bound at degree 0: log2(exp(width) * width) + (p - 1)
    bits_lost = (
        width * pymath.log2(pymath.e) + pymath.log2(width) + (precision_bits - 1)
    )
    while bits_lost > max_bits_lost and degree < _MAX_DEGREE:
        degree += 1
        # multiplier bound(d+1) / bound(d) = width / (4 * (degree + 1))
        # in log2 space:                   = log2(width) - 2 - log2(degree + 1)
        bits_lost += pymath.log2(width) - 2 - pymath.log2(degree + 1)

    return _chebyshev_coefficients(pymath.exp, degree, lower, upper)


class LowerExpToPolynomial(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        elem_ty = _element_float_type(op.operand.type)
        underflow = UNDERFLOW_LOWER_BOUND.get(type(elem_ty))
        overflow = OVERFLOW_UPPER_BOUND.get(type(elem_ty))
        precision_bits = _PRECISION_BITS.get(type(elem_ty))
        if underflow is None or overflow is None or precision_bits is None:
            return

        # If no acc bound, compute correctly-rounded result
        max_bits_lost_attr = op.attributes.get("max_bits_lost")
        max_bits_lost = (
            max_bits_lost_attr.value.data
            if isinstance(max_bits_lost_attr, FloatAttr)
            else -1.0
        )

        # Polynomial domain: each side independently clamped to its representable
        # extreme. Missing lower_bound -> underflow; missing upper_bound -> overflow.
        lower_attr = op.attributes.get("lower_bound")
        upper_attr = op.attributes.get("upper_bound")
        lower = (
            max(underflow, lower_attr.value.data)
            if isinstance(lower_attr, FloatAttr)
            else underflow
        )
        upper = (
            min(overflow, upper_attr.value.data)
            if isinstance(upper_attr, FloatAttr)
            else overflow
        )

        # choose which polynomial family and degree to use, and compute coefficients
        coeffs = _choose_polynomial(max_bits_lost, lower, upper, precision_bits)

        # insert polynomial into the IR replacing exp
        rewriter.replace_op(
            op,
            polynomial.EvalOp(
                value=op.operand,
                polynomial=tuple(coeffs),
                scheme=polynomial.EvalScheme.CLENSHAW,
                domain_lower=lower,
                domain_upper=upper,
            ),
        )


@dataclass(frozen=True)
class LowerExpToPolynomialPass(ModulePass):
    """Lower `math.exp` to `polynomial.eval`."""

    name = "lower-exp-to-polynomial"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerExpToPolynomial()).rewrite_module(op)
