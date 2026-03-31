from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, Region, SSAValue
from xdsl.ir.post_order import PostOrderIterator
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriterListener,
    PatternRewriteWalker,
    RewritePattern,
)
from xdsl.traits import (
    IsTerminator,
    MemoryEffectKind,
    SymbolOpInterface,
    get_effects,
)


def would_be_trivially_dead(op: Operation):
    """
    Returns if the operation would be dead if all its results were dead.
    """
    return (
        not op.has_trait(IsTerminator, value_if_unregistered=False)
        and (not op.has_trait(SymbolOpInterface, value_if_unregistered=False))
        and result_only_effects(op)
    )


def is_trivially_dead(op: Operation):
    """
    Returns if the operation has no observable effect.
    """
    return all(
        result.first_use is None for result in op.results
    ) and would_be_trivially_dead(op)


def result_only_effects(rootOp: Operation) -> bool:
    """
    Returns if we can ensure the operation would have no observable effect beyond its
    returned values.

    cf MLIR's WouldOpBeTriviallyDead:
    https://mlir.llvm.org/doxygen/namespacemlir.html#a655db45ed8c23d04d5ed5ee0abe041ad

    We have one key difference here:
    - MLIR discard any allocation from an operation on its own result for this analysis
    - xDSL discard any allocation effect of any nested operation on any value defined
    by the root operation or its children.
    """
    effects = get_effects(rootOp)
    # If the operation has unknown effect, we safely assume it has observable ones
    return effects is not None and all(
        # Read-only effect will not affect other operations
        e.kind == MemoryEffectKind.READ
        # Allocation of values defined by this operation or its children will not
        # affect other operations
        or (
            e.kind == MemoryEffectKind.ALLOC
            and isinstance(v := e.value, SSAValue)
            and rootOp.is_ancestor(v.owner)
        )
        for e in effects
    )


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


@dataclass
class LiveSet:
    changed: bool = field(default=True)  # Force first iteration to happen
    _live_ops: set[Operation] = field(default_factory=set[Operation])

    def is_live(self, op: Operation) -> bool:
        return op in self._live_ops

    def set_live(self, op: Operation):
        if not self.is_live(op):
            self.changed = True
            self._live_ops.add(op)

    def propagate_op_liveness(self, op: Operation):
        for region in op.regions:
            self.propagate_region_liveness(region)

        if self.is_live(op):
            return

        if not would_be_trivially_dead(op):
            self.set_live(op)
            return

        if any(
            self.is_live(use.operation) for result in op.results for use in result.uses
        ):
            self.set_live(op)

    def propagate_region_liveness(self, region: Region):
        first = region.first_block
        if first is None:
            return
        for block in PostOrderIterator(first):
            # Process operations in reverse order to speed convergence
            for operation in reversed(block.ops):
                self.propagate_op_liveness(operation)

    def delete_dead(self, region: Region, listener: PatternRewriterListener | None):
        first = region.first_block
        if first is None:
            return

        for block in reversed(region.blocks):
            if not any(self.is_live(op) for op in block.ops) and block != first:
                # If block is not the entry block and has no live ops then delete it
                self.changed = True
                region.erase_block(block, safe_erase=False)
                continue

            for operation in reversed(block.ops):
                if not self.is_live(operation):
                    self.changed = True
                    if listener is not None:
                        listener.handle_operation_removal(operation)
                    block.erase_op(operation, safe_erase=False)
                else:
                    for r in operation.regions:
                        self.delete_dead(r, listener)


def region_dce(region: Region, listener: PatternRewriterListener | None = None) -> bool:
    live_set = LiveSet()

    while live_set.changed:
        live_set.changed = False
        live_set.propagate_region_liveness(region)

    live_set.delete_dead(region, listener)
    return live_set.changed


class DeadCodeElimination(ModulePass):
    name = "dce"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        region_dce(op.body)
