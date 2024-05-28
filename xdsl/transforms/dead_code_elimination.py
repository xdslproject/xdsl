from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.traits import IsTerminator, SymbolOpInterface, is_side_effect_free


def is_trivially_dead(op: Operation):
    # Check that operation is side-effect-free and unused
    return (
        not op.get_trait(IsTerminator)
        and not op.get_trait(SymbolOpInterface)
        and is_side_effect_free(op)
        and all(not result.uses for result in op.results)
    )


class RemoveUnusedOperations(RewritePattern):
    """
    Removes operations annotated with the `Pure` trait, where results have no uses.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if is_trivially_dead(op):
            rewriter.erase_op(op)


def dce(op: ModuleOp):
    """
    Removes operations annotated with the `Pure` trait, where results have no uses.
    Modifies input module in-place.
    """
    walker = PatternRewriteWalker(
        RemoveUnusedOperations(), apply_recursively=True, walk_reverse=True
    )
    walker.rewrite_module(op)


class DeadCodeElimination(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        dce(op)
