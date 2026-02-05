from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, math
from xdsl.dialects.builtin import Float64Type, FloatAttr, ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException

f64 = Float64Type()


@dataclass
class ExpandExp(RewritePattern):
    """
    Replace `math.exp` operations with a polynomial expansion.
    """

    terms: int
    """Number of terms to use when expanding `math.exp`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        if op.operands[0].type != f64:
            raise PassFailedException("Expansion implemented only for f64.")

        expanded: Operation = expand_exp(op, rewriter, self.terms)
        rewriter.replace_op(op, (), (expanded.results[0],))


def expand_exp(op: math.ExpOp, rewriter: PatternRewriter, terms: int) -> Operation:
    """
    Expand exp(x) using the Taylor-series loop from the pseudo-code:

        terms = 75
        result = 1.0
        term = 1.0
        for i in range(1, terms): # loop will be unrolled by the rewriter
            term *= x / i
            result += term
        return result
    """
    x = op.operands[0]

    res = rewriter.insert(arith.ConstantOp(FloatAttr(1.0, f64)))
    term = rewriter.insert(arith.ConstantOp(FloatAttr(1.0, f64)))

    for i in range(1, terms):
        i_val = rewriter.insert(arith.ConstantOp(FloatAttr(float(i), f64)))
        frac = rewriter.insert(arith.DivfOp(x, i_val.result))
        mul = rewriter.insert(arith.MulfOp(frac.result, term.result))
        add = rewriter.insert(arith.AddfOp(res.result, mul.result))

        term = mul
        res = add

    return res


@dataclass(frozen=True)
class MathToPolynomialsPass(ModulePass):
    """This pass expands `math.exp` operations to a polynomial expansion using the Taylor series."""

    name = "expand-math-to-polynomials"

    terms = 75
    """Number of terms to use when expanding `math.exp`."""

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandExp(self.terms),
            apply_recursively=False,
        ).rewrite_module(op)
