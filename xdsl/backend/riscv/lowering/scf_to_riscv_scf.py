from xdsl.backend.riscv.lowering.utils import (
    cast_operands_to_int_regs,
    cast_results_to_int_regs,
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
        lb, ub, step, *args = cast_operands_to_int_regs(
            [op.lb, op.ub, op.step, *op.iter_args], rewriter
        )
        for_op = riscv_scf.ForOp(
            lb, ub, step, args, rewriter.move_region_contents_to_new_regions(op.body)
        )
        cast_results_to_int_regs(op.results, rewriter)
        rewriter.replace_matched_op([for_op])


class ScfYieldLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.Yield, rewriter: PatternRewriter) -> None:
        args_cast = cast_operands_to_int_regs(op.arguments, rewriter)
        yield_op = riscv_scf.YieldOp(*args_cast)
        rewriter.replace_matched_op([yield_op])


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
