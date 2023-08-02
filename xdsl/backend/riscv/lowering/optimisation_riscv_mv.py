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
from xdsl.transforms.dead_code_elimination import dce


class RemoveMv(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MVOp, rewriter: PatternRewriter) -> None:
        assert isinstance(op.rd.type, riscv.IntRegisterType), op.rd.type
        assert isinstance(op.rs.type, riscv.IntRegisterType), op.rs.type
        if op.rd.type.name == op.rs.type.name:
            rewriter.replace_matched_op([], [op.rs])


class RISCVMvOpt(ModulePass):
    name = "optimisation-riscv-mv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveMv(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)

        dce(op)
