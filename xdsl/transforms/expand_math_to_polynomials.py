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


@dataclass
class ExpandExp(RewritePattern):
    """
    Replace `math.exp` operations with a polynomial expansion.

    Only expands when the number of terms is specified, either via an
    attribute on the operation or via the pass-level default.
    """

    default_terms: int | None = None
    """Pass-level default for number of terms. None means don't expand
    unless the operation has an explicit terms attribute."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        terms: int | None = None
        if "terms" in op.attributes:
            attr = op.attributes["terms"]
            if isinstance(attr, IntegerAttr):
                terms = attr.value.data
        elif self.default_terms is not None:
            terms = self.default_terms

        if terms is None:
            return

        expanded: Operation = expand_exp(op, rewriter, terms)
        rewriter.replace_op(op, (), (expanded.results[0],))


def _float_constant(
    value: float,
    tp: AnyFloat | VectorType[AnyFloat] | TensorType[AnyFloat],
    rewriter: PatternRewriter,
) -> arith.ConstantOp:
    """Create and insert a float constant (arith.ConstantOp) for a given float value, handling both scalar and vector types."""
    if isa(tp, VectorType[AnyFloat]):
        attr = DenseIntOrFPElementsAttr.from_list(tp, [value])
    elif isa(tp, TensorType[AnyFloat]):
        attr = DenseIntOrFPElementsAttr.from_list(tp, [value])
    elif isa(tp, AnyFloat):
        attr = FloatAttr(value, tp)
    else:
        raise TypeError(f"Unsupported type for float constant: {tp}")
    return rewriter.insert(arith.ConstantOp(attr))


def expand_exp(op: math.ExpOp, rewriter: PatternRewriter, terms: int) -> Operation:
    """
    Expand exp(x) using a Taylor-series polynomial expansion.

    Pseudo-code::

        result = 1.0
        term = 1.0
        for i in range(1, terms): # loop will be unrolled by the rewriter
            term *= x / i
            result += term
        return result
    """
    x = op.operands[0]
    tp = x.type
    if not isa(tp, AnyFloat | VectorType[AnyFloat] | TensorType[AnyFloat]):
        raise TypeError(f"Unsupported type for math.exp expansion: {tp}")

    res = _float_constant(1.0, tp, rewriter)
    term = _float_constant(1.0, tp, rewriter)

    for i in range(1, terms):
        i_val = _float_constant(1.0 / float(i), tp, rewriter)
        frac = rewriter.insert(arith.MulfOp(x, i_val.result))
        mul = rewriter.insert(arith.MulfOp(frac.result, term.result))
        add = rewriter.insert(arith.AddfOp(res.result, mul.result))

        term = mul
        res = add

    return res


@dataclass(frozen=True)
class ExpandMathToPolynomialsPass(ModulePass):
    """
    This pass expands `math` operations to a polynomial expansion using the Taylor series.

    Currently only expands `math.exp` operations.

    Operations are only expanded when the number of terms is specified,
    either via a `terms` attribute on the operation itself or via the
    pass-level `terms` parameter.
    """

    name = "expand-math-to-polynomials"

    terms: int | None = None
    """Number of terms in the resulting polynomial expansion.
    If not set, only operations with an explicit terms attribute are expanded."""

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandExp(default_terms=self.terms),
            apply_recursively=False,
        ).rewrite_module(op)
