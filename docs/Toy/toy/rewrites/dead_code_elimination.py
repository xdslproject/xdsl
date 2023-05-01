from xdsl.ir import OpTrait, Operation
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter


class Pure(OpTrait):
    """A trait that signals that an operation has no side effects."""


class RemoveUnusedOperations(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        """
        Removes operations whose result is not used, and that don't have side effects
        """
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
