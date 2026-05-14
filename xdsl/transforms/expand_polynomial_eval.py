"""
Expand `polynomial.eval` ops to arithmetic operations.

This pass dispatches on the `scheme` attribute and emits the corresponding arith ops.

Currently supported schemes:
  - "clenshaw": Chebyshev series evaluated via Clenshaw's recurrence.

"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, polynomial
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    ModuleOp,
    TensorType,
    VectorType,
)
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import isa
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


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


def expand_clenshaw(
    op: polynomial.EvalOp,
    rewriter: PatternRewriter,
    coeffs: tuple[float],
    lower: float | None,
    upper: float | None,
) -> Operation:
    """
    Expand a Chebyshev series via Clenshaw's recurrence into arith ops.

    If no domain bounds are provided, the input is assumed to already be
    in the canonical Chebyshev domain [-1, 1].

    Returns the final operation whose result is the evaluation result.
    """
    x = op.value
    tp = x.type
    if not isa(tp, AnyFloat | VectorType[AnyFloat] | TensorType[AnyFloat]):
        raise TypeError(f"Unsupported type for polynomial.eval expansion: {tp}")

    n = len(coeffs) - 1

    # --- domain mapping: t = x * scale + offset --------------------------
    t: SSAValue
    if lower is not None and upper is not None:
        scale = 2.0 / (upper - lower)
        offset = -(upper + lower) / (upper - lower)
        scale_op = _float_constant(scale, tp, rewriter)
        offset_op = _float_constant(offset, tp, rewriter)
        scaled = rewriter.insert(arith.MulfOp(x, scale_op.result))
        t = rewriter.insert(arith.AddfOp(scaled.result, offset_op.result)).result
    else:
        t = x  # canonical Chebyshev domain [-1, 1]

    # --- Clenshaw recurrence --------------------------------------------
    two = _float_constant(2.0, tp, rewriter)
    two_t = rewriter.insert(arith.MulfOp(two.result, t))

    b_prev2 = _float_constant(0.0, tp, rewriter)  # b_{n+2}
    b_prev1 = _float_constant(0.0, tp, rewriter)  # b_{n+1}

    for k in range(n, 0, -1):
        c_k = _float_constant(coeffs[k], tp, rewriter)
        two_t_b = rewriter.insert(arith.MulfOp(two_t.result, b_prev1.result))
        sub = rewriter.insert(arith.SubfOp(two_t_b.result, b_prev2.result))
        b_k = rewriter.insert(arith.AddfOp(sub.result, c_k.result))
        b_prev2 = b_prev1
        b_prev1 = b_k

    # --- final: result = c_0/2 + t * b_1 - b_2 ---------------------------
    c0_half = _float_constant(coeffs[0] / 2.0, tp, rewriter)
    t_b1 = rewriter.insert(arith.MulfOp(t, b_prev1.result))
    add = rewriter.insert(arith.AddfOp(c0_half.result, t_b1.result))
    return rewriter.insert(arith.SubfOp(add.result, b_prev2.result))


class ExpandPolynomialEval(RewritePattern):
    """Replace each `polynomial.eval` op with the arith ops for its scheme."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: polynomial.EvalOp, rewriter: PatternRewriter
    ) -> None:
        coeffs = op.polynomial.coeff_values
        lower = op.domain_lower.value.data if op.domain_lower is not None else None
        upper = op.domain_upper.value.data if op.domain_upper is not None else None

        scheme = op.eval_scheme
        if scheme is polynomial.EvalScheme.CLENSHAW:
            expanded = expand_clenshaw(op, rewriter, coeffs, lower, upper)
        else:
            # Verifier already restricts `scheme` to known EvalScheme members,
            # so this is only reachable if a new scheme is added without a
            # corresponding lowering branch.
            raise NotImplementedError(
                f"polynomial.eval scheme {scheme.value!r} has no lowering"
            )

        rewriter.replace_op(op, (), (expanded.results[0],))


@dataclass(frozen=True)
class ExpandPolynomialEvalPass(ModulePass):
    """
    Expand `polynomial.eval` ops to arithmetic operations.

    All information needed for lowering (coefficients, scheme, domain
    bounds) is read directly from each op, so this pass takes no
    parameters.
    """

    name = "expand-polynomial-eval"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandPolynomialEval(),
            apply_recursively=False,
        ).rewrite_module(op)
