from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpTrait, Operation
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, PatternRewriteWalker


class Pure(OpTrait):
    """A trait that signals that an operation has no side effects."""


class RemoveUnusedOperations(RewritePattern):
    """
    Removes operations annotated with the `Pure` trait, where results have no uses.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Check that operation is side-effect-free
        if not op.has_trait(Pure()):
            return

        # Look through the input of the current transpose.
        results = op.results
        for result in results:
            if len(result.uses):
                # At least one of the results is used
                return

        rewriter.erase_op(op)


def dce(module: ModuleOp):
    """
    Rewrites the module in-place to remove trivially unused operations.
    """
    PatternRewriteWalker(
        RemoveUnusedOperations(), apply_recursively=True, walk_reverse=True
    ).rewrite_module(module)
    pass
