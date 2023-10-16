from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
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
                new_lb := riscv.MulOp(
                    op.lb, factor, rd=riscv.IntRegisterType.unallocated()
                ),
                new_ub := riscv.MulOp(
                    op.ub, factor, rd=riscv.IntRegisterType.unallocated()
                ),
                new_step := riscv.MulOp(
                    op.step, factor, rd=riscv.IntRegisterType.unallocated()
                ),
            ]
        )

        op.operands[0] = new_lb.rd
        op.operands[1] = new_ub.rd
        op.operands[2] = new_step.rd

        for mul_op in uses:
            rewriter.replace_op(mul_op, [], [index])


class RiscvScfLoopRangeFoldingPass(ModulePass):
    """
    Similar to scf-loop-range-folding in MLIR, folds multiplication operations into the
    loop range computation when possible.
    """

    name = "riscv-scf-loop-range-folding"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            HoistIndexTimesConstant(),
            apply_recursively=False,
        ).rewrite_module(op)
