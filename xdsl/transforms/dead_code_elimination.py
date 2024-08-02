from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.traits import (
    IsTerminator,
    MemoryEffectKind,
    SymbolOpInterface,
    get_effects,
)


def is_trivially_dead(op: Operation):
    """
    Returns if the operation has no observable effect.
    """
    return (
        all(not result.uses for result in op.results)
        and (not op.get_trait(IsTerminator))
        and (not op.get_trait(SymbolOpInterface))
        and result_only_effects(op)
    )


def result_only_effects(rootOp: Operation) -> bool:
    """
    Returns if we can ensure the operation would have no observable effect beyond its
    returned values.

    cf MLIR's WouldOpBeTriviallyDead:
    https://mlir.llvm.org/doxygen/namespacemlir.html#a655db45ed8c23d04d5ed5ee0abe041ad
    """
    effects = get_effects(rootOp)
    return effects is not None and all(e == MemoryEffectKind.READ for e in effects)


class RemoveUnusedOperations(RewritePattern):
    """
    Removes operations annotated with the `Pure` trait, where results have no uses.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if is_trivially_dead(op) and op.parent is not None:
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
