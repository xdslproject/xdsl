from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir.core import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveRedundantMv(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MVOp, rewriter: PatternRewriter) -> None:
        if op.rd.type == op.rs.type:
            rewriter.replace_matched_op([], [op.rs])


class OptimiseRiscvPass(ModulePass):
    name = "optimise-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveRedundantMv(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
