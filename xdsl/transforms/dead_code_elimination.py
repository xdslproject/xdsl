from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, PatternRewriteWalker
from xdsl.traits import Pure


class RemoveUnusedOperations(RewritePattern):
    """
    Removes operations annotated with the `Pure` trait, where results have no uses.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Check that operation is side-effect-free
        if not op.has_trait(Pure()):
            return

        # Check whether any of the results are used
        results = op.results
        for result in results:
            if len(result.uses):
                # At least one of the results is used
                return

        rewriter.erase_op(op)


class DeadCodeElimination(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            RemoveUnusedOperations(), apply_recursively=True, walk_reverse=True
        )
        walker.rewrite_module(op)
