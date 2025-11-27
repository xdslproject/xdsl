from xdsl.backend.riscv.lowering.utils import (
    cast_block_args_to_regs,
    cast_matched_op_results,
    cast_operands_to_regs,
    move_to_unallocated_regs,
)
from xdsl.context import Context
from xdsl.dialects import builtin, riscv_scf, scf
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
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        lb, ub, step, *args = cast_operands_to_regs(rewriter)
        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        cast_block_args_to_regs(new_region.block, rewriter)
        mv_ops, values = move_to_unallocated_regs(args, op.iter_args.types)
        rewriter.insert_op(mv_ops)
        cast_matched_op_results(rewriter)
        new_op = riscv_scf.ForOp(lb, ub, step, values, new_region)
        mv_res_ops, res_values = move_to_unallocated_regs(
            new_op.results, op.iter_args.types
        )

        rewriter.replace_op(op, (new_op, *mv_res_ops), res_values)


class ScfYieldLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.YieldOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, riscv_scf.YieldOp(*cast_operands_to_regs(rewriter)))


class ConvertScfToRiscvPass(ModulePass):
    name = "convert-scf-to-riscv-scf"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ScfYieldLowering(),
                    ScfForLowering(),
                ]
            )
        ).rewrite_module(op)
