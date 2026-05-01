import math as pymath
from collections.abc import Callable
from dataclasses import dataclass

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

_UNDERFLOW_LOWER_BOUND: dict[type, float] = {
    # Float16Type: pymath.log(2.0**-24),  # ≈ -16.64
    # BFloat16Type: pymath.log(2.0**-133),  # ≈ -92.18
    # Float32Type: pymath.log(2.0**-149),  # ≈ -103.28
    # Float64Type: pymath.log(2.0**-1074),  # ≈ -744.44
    Float16Type: -1,
    BFloat16Type: -1,
    Float32Type: -1,
    Float64Type: -1,
}

_DEFAULT_DEGREE = 10


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
        return tp.get_element_type()
    return tp


def _choose_polynomial(
    acc_bound: float | None,
    lower: float,
    upper: float,
) -> list[float]:
    """Smallest degree such that the Lobatto Chebyshev bound for exp on [a,b] is <= acc_bound."""
    if acc_bound is None:
        return _chebyshev_coefficients(pymath.exp, _DEFAULT_DEGREE, lower, upper)

    width = upper - lower
    degree = 0
    bound = pymath.exp(upper) * width
    while bound > acc_bound:
        degree += 1
        bound *= width / (4 * (degree + 1))

    return _chebyshev_coefficients(pymath.exp, degree, lower, upper)


class LowerExpToPolynomial(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        # derive lower and upper bounds from precision format
        # this is specific to exp
        elem_ty = _element_float_type(op.operand.type)
        lower = _UNDERFLOW_LOWER_BOUND.get(type(elem_ty))
        if lower is None:
            return
        upper = 0.0

        # accuracy bound carried over from softmax (step 1); drives the
        # polynomial degree the chooser picks. Absent => no constraint.
        acc_bound_attr = op.attributes.get("acc_bound")
        acc_bound = (
            acc_bound_attr.value.data if isinstance(acc_bound_attr, FloatAttr) else None
        )

        # choose which polynomial family and degree to use, and compute coefficients
        coeffs = _choose_polynomial(acc_bound, lower, upper)

        # insert polynomial into the IR replacing exp
        rewriter.replace_matched_op(
            polynomial.EvalOp(
                value=op.operand,
                polynomial=tuple(coeffs),
                scheme=polynomial.EvalScheme.CLENSHAW,
                domain_lower=lower,
                domain_upper=upper,
            )
        )


@dataclass(frozen=True)
class LowerExpToPolynomialPass(ModulePass):
    """Lower `math.exp` to `polynomial.eval`."""

    name = "lower-exp-to-polynomial"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerExpToPolynomial()).rewrite_module(op)
