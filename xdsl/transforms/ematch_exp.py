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

f64 = Float64Type()


class ExpandExp(RewritePattern):
    """
    Replace `exp` operations with a polynomial expansion.
    """

    def __init__(self, terms: int):
        self.terms = terms

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        if op.operands[0].type != f64:
            return  # only handle f64 for now

        expanded: Operation = expand_exp(op, rewriter, self.terms)

        rewriter.replace_op(
            op, expanded, ()
        )  # replace will create an equivalence class to union op and expanded internally


def expand_exp(op: math.ExpOp, rewriter: PatternRewriter, terms: int) -> Operation:
    """
    Expand exp(x) using the Taylor-series loop from the C code:

        terms = 75
        result = 1.0
        term = 1.0
        for i = 1 .. terms-1:
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
class EmatchExpPass(ModulePass):
    """
    A pass that expands `math` operations to a Taylor series polynomial expansion equality saturation.

    Currently only expands `math.exp` operations.
    """

    name = "expand-exp-to-polynomials"
    terms = 75

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandExp(self.terms),
            apply_recursively=False,
            # we want to use the equivalence rewriter
            eq_rewriter=True,
        ).rewrite_module(op)
