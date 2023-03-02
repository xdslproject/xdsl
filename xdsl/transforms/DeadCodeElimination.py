from xdsl.traits import Pure
from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, Operation
from xdsl.dialects.builtin import ModuleOp


class UnusedOperationRemover(RewritePattern):

    traits = (Pure, )

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if all(len(res.uses) == 0 for res in op.results):
            rewriter.erase_matched_op()


def dce(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier(
        [UnusedOperationRemover()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
