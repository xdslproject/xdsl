from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.traits import (
    EffectKind,
    IsTerminator,
    MemoryEffect,
    RecursiveMemoryEffect,
    SymbolOpInterface,
)


def is_trivially_dead(op: Operation):
    # Check that operation is side-effect-free and unused
    return (
        all(not result.uses for result in op.results)
        and (not op.get_trait(IsTerminator))
        and (not op.get_trait(SymbolOpInterface))
        and would_be_trivially_dead(op)
    )


def would_be_trivially_dead(rootOp: Operation) -> bool:
    effecting_ops = {rootOp}
    while effecting_ops:
        op = effecting_ops.pop()

        # If the operation has recursive effects, push all of the nested operations
        # on to the stack to consider.
        recursive = op.get_trait(RecursiveMemoryEffect)
        if recursive:
            effecting_ops.update(o for r in op.regions for b in r.blocks for o in b.ops)

        if effect_interface := op.get_trait(MemoryEffect):
            effects = effect_interface.get_effects(op)

            # Currently, only read effects are considered potentially dead.
            # MLIR does smarter things with allocated values here.
            if any(e != EffectKind.READ for e in effects):
                return False

            continue

        if recursive:
            continue

        return False
    return True


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
