from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class HoistIndexTimesConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter) -> None:
        index = op.body.block.args[0]

        uses: list[riscv.MulOp] = []
        constant: int | None = None
        for use in index.uses:
            if not isinstance(mul_op := use.operation, riscv.MulOp):
                # One of the uses is not a multiplication, bail
                return

            uses.append(mul_op)

            if mul_op.rs1 is index:
                if not isinstance(li_op := mul_op.rs2.owner, riscv.LiOp):
                    # One of the uses is not a multiplication by constant, bail
                    return
            else:
                if not isinstance(li_op := mul_op.rs1.owner, riscv.LiOp):
                    # One of the uses is not a multiplication by constant, bail
                    return

            if not isinstance(imm := li_op.immediate, builtin.IntegerAttr):
                # One of the constants is not an integer, bail
                return

            if constant is None:
                constant = imm.value.data
            else:
                if constant != imm.value.data:
                    # Not all constants are equal, bail
                    return

        if constant is None:
            # No uses, bail
            return

        # All the uses are multiplications by a constant, we can fold
        rewriter.insert_op_before_matched_op(
            [
                factor := riscv.LiOp(constant),
                old_distance := riscv.SubOp(op.ub, op.lb),
                new_distance := riscv.MulOp(old_distance, factor),
                new_ub := riscv.AddOp(op.lb, new_distance),
                new_step := riscv.MulOp(op.step, factor),
            ]
        )

        op.operands[1] = new_ub.rd
        op.operands[2] = new_step.rd

        for mul_op in uses:
            rewriter.replace_op(mul_op, [], [index])


class RISCVOptimize(ModulePass):
    name = "riscv-optimize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    HoistIndexTimesConstant(),
                ]
            )
        ).rewrite_module(op)
