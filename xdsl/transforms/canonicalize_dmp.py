from xdsl.dialects import builtin
from xdsl.dialects.experimental import dmp
from xdsl.passes import MLContext, ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class CanonicalizeDmpSwap(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        keeps: list[dmp.ExchangeDeclarationAttr] = []
        for swap in op.swaps:
            if swap.elem_count > 0:
                keeps.append(swap)
        if len(keeps) == 0:
            rewriter.erase_matched_op()
        else:
            op.swaps = builtin.ArrayAttr(keeps)


class CanonicalizeDmpPass(ModulePass):
    name = "canonicalize-dmp"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(CanonicalizeDmpSwap()).rewrite_module(op)
