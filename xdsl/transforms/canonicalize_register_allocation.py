from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.transforms.canonicalization_patterns.riscv import (
    RemoveRedundantFMv,
    RemoveRedundantFMvD,
    RemoveRedundantMv,
)
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations, region_dce


class CanonicalizePostRegisterAllocationPass(ModulePass):
    """
    Removes redundant moves and unused operations after register allocation.
    """

    name = "canonicalize-register-allocation"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveUnusedOperations(),
                    RemoveRedundantMv(),
                    RemoveRedundantFMv(),
                    RemoveRedundantFMvD(),
                ]
            ),
            post_walk_func=region_dce,
        ).rewrite_module(op)
