from xdsl.traits import NoSideEffect
from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, Operation
from xdsl.dialects.builtin import ModuleOp


class DCEImpl(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if isinstance(op, NoSideEffect) and all(
            [len(res.uses) == 0 for res in op.results]):
            rewriter.erase_matched_op()


def dce(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([DCEImpl()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
