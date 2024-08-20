from xdsl.dialects import builtin, stencil
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
            new_result = (
                op.input_stencil
                if isinstance(op.input_stencil.type, stencil.TempType)
                else None
            )
            rewriter.replace_matched_op([], [new_result])
        else:
            op.swaps = builtin.ArrayAttr(keeps)


class CanonicalizeDmpPass(ModulePass):
    name = "canonicalize-dmp"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(CanonicalizeDmpSwap()).rewrite_module(op)
