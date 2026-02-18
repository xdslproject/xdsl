from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, math
from xdsl.dialects.builtin import Float64Type, FloatAttr, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.pattern_rewriter_eq import EquivalencePatternRewriter

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
    Expand exp(x) using the Taylor-series loop from the pseudo code:

        terms = 75
        result = 1.0
        term = 1.0
        for i in range(1, terms):
            term *= x / i
            result += term
        return result
    """

    def gcr(val: SSAValue) -> SSAValue:
        """
        If EquivalencePatternRewriter ist used get the class result for a value,
        with a non-optional return type, else do nothing
        """
        if isinstance(rewriter, EquivalencePatternRewriter):
            result = rewriter.eqsat_bookkeeping.run_get_class_result(val)
            if result is not None:
                return result

        return val

    x = op.operands[0]

    res = rewriter.insert(arith.ConstantOp(FloatAttr(1.0, f64)))
    term = rewriter.insert(arith.ConstantOp(FloatAttr(1.0, f64)))

    for i in range(1, terms):
        i_val = rewriter.insert(arith.ConstantOp(FloatAttr(float(i), f64)))
        frac = rewriter.insert(arith.DivfOp(gcr(x), gcr(i_val.result)))
        mul = rewriter.insert(arith.MulfOp(gcr(frac.result), gcr(term.result)))
        add = rewriter.insert(arith.AddfOp(gcr(res.result), gcr(mul.result)))

        term = mul
        res = add

    return res


@dataclass(frozen=True)
class EmatchExpPass(ModulePass):
    """
    Matches `math.exp` operations and adds their Taylor series polynomial
    expansion as equivalent representations in the e-graph.
    """

    name = "ematch-exp"
    terms = 3

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ExpandExp(self.terms),
            apply_recursively=False,
            rewriter_factory=EquivalencePatternRewriter,  # we want to use the equivalence rewriter
        ).rewrite_module(op)
