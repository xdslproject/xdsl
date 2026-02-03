from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, math
from xdsl.dialects.builtin import Float64Type, FloatAttr, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class ExpToTaylorPass(ModulePass):
    name = "convert-exp-to-taylor"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ExpandExp()]),
            apply_recursively=False,
        ).rewrite_module(op)


class ExpandExp(RewritePattern):
    """
    Replace `exp` operations with a polynomial expansion.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpOp, rewriter: PatternRewriter) -> None:
        if op.operands[0].type != Float64Type():
            return  # only handle f64 for now

        builder = Builder(InsertPoint.before(op))

        expanded: Operation = expand_exp(op, builder)
        rewriter.replace_op(op, expanded)


def expand_exp(op: math.ExpOp, builder: Builder) -> Operation:
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

    x: SSAValue = op.operands[0]

    res = builder.insert(arith.ConstantOp(FloatAttr(1.0, Float64Type())))
    term = builder.insert(arith.ConstantOp(FloatAttr(1.0, Float64Type())))

    terms = 75
    replacement: Operation | None = None

    for i in range(1, terms):
        i_val = builder.insert(arith.ConstantOp(FloatAttr(float(i), Float64Type())))
        frac = builder.insert(arith.DivfOp(x, i_val.result))
        mul = builder.insert(arith.MulfOp(frac.result, term.result))

        add = arith.AddfOp(res.result, mul.result)

        if i == terms - 1:
            # keep last add unattached -> returned to rewriter
            replacement = add
        else:
            builder.insert(add)
            term = mul
            res = add

    assert replacement is not None
    return replacement
