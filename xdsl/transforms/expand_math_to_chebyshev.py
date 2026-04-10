"""
Expand math.exp using Chebyshev polynomial approximation with Clenshaw evaluation.

This pass replaces math.exp operations with a near-minimax Chebyshev polynomial
approximation evaluated using Clenshaw's algorithm. Unlike the Taylor series
approach in expand_math_to_polynomials, this provides near-optimal accuracy for
a given polynomial degree on a bounded interval [lower, upper].

The approach is inspired by Google's HEIR compiler, which uses
Caratheodory-Fejer approximation with Chebyshev polynomials for FHE.

Algorithm:
  1. At compile time, compute Chebyshev coefficients c_0..c_n for exp(x)
     on [lower, upper] via DCT on Chebyshev nodes.
  2. At runtime (as arith ops):
     a. Map input x from [lower, upper] to t in [-1, 1].
     b. Evaluate the Chebyshev series using Clenshaw's recurrence:
          b_{n+1} = b_{n+2} = 0
          for k = n, ..., 1:  b_k = 2*t*b_{k+1} - b_{k+2} + c_k
          result = c_0/2 + t*b_1 - b_2
"""

import math as pymath
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, math
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    TensorType,
    VectorType,
)
from xdsl.ir import Operation
from xdsl.irdl import isa
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def chebyshev_coefficients(
    f: "Callable[[float], float]",  # noqa: F821
    degree: int,
    lower: float = -1.0,
    upper: float = 1.0,
) -> list[float]:
    """
    Compute Chebyshev coefficients c_0..c_n for f on [lower, upper].

    Uses DCT-I on n+1 Chebyshev nodes.  The expansion convention is

        f(x) ≈ c_0/2 + Σ_{k=1}^{n} c_k · T_k(t)

    where t = (2x − lower − upper) / (upper − lower) maps [lower, upper]
    to [−1, 1].
    """
    n = degree
    # Chebyshev nodes on [-1, 1]
    nodes = [pymath.cos(pymath.pi * j / n) for j in range(n + 1)]
    # Map to [lower, upper] and evaluate f
    mid = (upper + lower) / 2.0
    half = (upper - lower) / 2.0
    values = [f(half * t + mid) for t in nodes]
    # DCT-I: c_k = (2/n) * Σ'' f(x_j) cos(k j π / n)
    coeffs: list[float] = []
    for k in range(n + 1):
        s = 0.0
        for j in range(n + 1):
            w = 0.5 if (j == 0 or j == n) else 1.0
            s += w * values[j] * pymath.cos(pymath.pi * k * j / n)
        coeffs.append(2.0 * s / n)
    return coeffs


def _float_constant(
    value: float,
    tp: AnyFloat | VectorType[AnyFloat] | TensorType[AnyFloat],
    rewriter: PatternRewriter,
) -> arith.ConstantOp:
    """Create and insert a float constant, handling scalar/vector/tensor types."""
    if isa(tp, VectorType[AnyFloat]):
        attr = DenseIntOrFPElementsAttr.from_list(tp, [value])
    elif isa(tp, TensorType[AnyFloat]):
        attr = DenseIntOrFPElementsAttr.from_list(tp, [value])
    elif isa(tp, AnyFloat):
        attr = FloatAttr(value, tp)
    else:
        raise TypeError(f"Unsupported type for float constant: {tp}")
    return rewriter.insert(arith.ConstantOp(attr))


def expand_exp_chebyshev(
    op: math.ExpOp,
    rewriter: PatternRewriter,
    degree: int,
    lower: float,
    upper: float,
) -> Operation:
    """
    Expand exp(x) as a Chebyshev polynomial evaluated via Clenshaw's algorithm.

    Generated arith ops:
      1. Domain mapping:  t = x * scale + offset
      2. Precompute:      two_t = 2 * t
      3. Clenshaw loop (unrolled):
           b_{k} = two_t * b_{k+1} − b_{k+2} + c_k   (k = n … 1)
      4. Final:           result = c_0/2 + t * b_1 − b_2
    """
    x = op.operands[0]
    tp = x.type
    if not isa(tp, AnyFloat | VectorType[AnyFloat] | TensorType[AnyFloat]):
        raise TypeError(f"Unsupported type for math.exp expansion: {tp}")

    # --- compile-time: Chebyshev coefficients ---
    coeffs = chebyshev_coefficients(pymath.exp, degree, lower, upper)

    # --- runtime: domain mapping  t = x * scale + offset ---
    scale = 2.0 / (upper - lower)
    offset = -(upper + lower) / (upper - lower)
    scale_op = _float_constant(scale, tp, rewriter)
    offset_op = _float_constant(offset, tp, rewriter)
    scaled = rewriter.insert(arith.MulfOp(x, scale_op.result))
    t = rewriter.insert(arith.AddfOp(scaled.result, offset_op.result))

    # --- runtime: Clenshaw recurrence ---
    two_t = rewriter.insert(
        arith.MulfOp(
            _float_constant(2.0, tp, rewriter).result,
            t.result,
        )
    )
    b_prev2 = _float_constant(0.0, tp, rewriter)  # b_{n+2}
    b_prev1 = _float_constant(0.0, tp, rewriter)  # b_{n+1}

    for k in range(degree, 0, -1):
        c_k = _float_constant(coeffs[k], tp, rewriter)
        two_t_b = rewriter.insert(arith.MulfOp(two_t.result, b_prev1.result))
        sub = rewriter.insert(arith.SubfOp(two_t_b.result, b_prev2.result))
        b_k = rewriter.insert(arith.AddfOp(sub.result, c_k.result))
        b_prev2 = b_prev1
        b_prev1 = b_k

    # --- final: result = c_0/2 + t * b_1 - b_2 ---
    c0_half = _float_constant(coeffs[0] / 2.0, tp, rewriter)
    t_b1 = rewriter.insert(arith.MulfOp(t.result, b_prev1.result))
    add = rewriter.insert(arith.AddfOp(c0_half.result, t_b1.result))
    result = rewriter.insert(arith.SubfOp(add.result, b_prev2.result))

    return result


@dataclass
class ExpandExpChebyshev(RewritePattern):
    """Replace ``math.exp`` with a Chebyshev polynomial approximation.

    Only expands when degree, lower, and upper are specified, either via
    attributes on the operation or via the pass-level defaults.
    """

    default_degree: int | None = None
    """Pass-level default for degree. None means don't expand
    unless the operation has an explicit degree attribute."""

    default_lower: float | None = None
    """Pass-level default for lower bound of the approximation interval."""

    default_upper: float | None = None
    """Pass-level default for upper bound of the approximation interval."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        degree: int | None = None
        if "degree" in op.attributes:
            attr = op.attributes["degree"]
            if isinstance(attr, IntegerAttr):
                degree = attr.value.data
        elif self.default_degree is not None:
            degree = self.default_degree

        if degree is None:
            return

        lower: float | None = None
        if "lower" in op.attributes:
            lower_attr = op.attributes["lower"]
            if isinstance(lower_attr, FloatAttr):
                lower = lower_attr.value.data
        elif self.default_lower is not None:
            lower = self.default_lower

        if lower is None:
            return

        upper: float | None = None
        if "upper" in op.attributes:
            upper_attr = op.attributes["upper"]
            if isinstance(upper_attr, FloatAttr):
                upper = upper_attr.value.data
        elif self.default_upper is not None:
            upper = self.default_upper

        if upper is None:
            return

        expanded = expand_exp_chebyshev(op, rewriter, degree, lower, upper)
        rewriter.replace_op(op, (), (expanded.results[0],))


@dataclass(frozen=True)
class ExpandMathToChebyshevPass(ModulePass):
    """
    Expand ``math.exp`` using Chebyshev polynomial approximation.

    Uses near-minimax Chebyshev approximation on [lower, upper] evaluated
    via Clenshaw's algorithm.  For a given polynomial degree this gives
    much better accuracy than a Taylor series on the same interval.
    """

    name = "expand-math-to-chebyshev"

    degree: int | None = None
    """Degree of the Chebyshev polynomial (higher = more accurate).
    If not set, only operations with an explicit degree attribute are expanded."""

    lower: float | None = None
    """Lower bound of the approximation interval.
    If not set, only operations with an explicit lower attribute are expanded."""

    upper: float | None = None
    """Upper bound of the approximation interval.
    If not set, only operations with an explicit upper attribute are expanded."""

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandExpChebyshev(
                default_degree=self.degree,
                default_lower=self.lower,
                default_upper=self.upper,
            ),
            apply_recursively=False,
        ).rewrite_module(op)
