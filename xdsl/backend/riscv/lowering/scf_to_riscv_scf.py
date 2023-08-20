from xdsl.backend.riscv.lowering.utils import (
    cast_block_args_to_regs,
    cast_matched_op_results,
    cast_operands_to_regs,
    move_ops_for_value,
)
from xdsl.dialects import builtin, riscv_scf, scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ScfForLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter) -> None:
        lb, ub, step, *args = cast_operands_to_regs(rewriter)
        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        cast_block_args_to_regs(new_region.block, rewriter)
        mv_ops = [move_ops_for_value(arg) for arg in args]
        rewriter.insert_op_before_matched_op([mv_op for ops in mv_ops for mv_op in ops])
        cast_matched_op_results(rewriter)
        rewriter.replace_matched_op(
            riscv_scf.ForOp(
                lb, ub, step, [mv_op[-1].rd for mv_op in mv_ops], new_region
            )
        )


class ScfYieldLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.Yield, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op(riscv_scf.YieldOp(*cast_operands_to_regs(rewriter)))


class ScfToRiscvPass(ModulePass):
    name = "scf-to-rvscf-lowering"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ScfYieldLowering(),
                    ScfForLowering(),
                ]
            )
        ).rewrite_module(op)
