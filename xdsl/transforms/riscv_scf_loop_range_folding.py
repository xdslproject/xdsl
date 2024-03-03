from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value


class HoistIndexTimesConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter) -> None:
        index = op.body.block.args[0]

        # Fold until a fixed point is reached
        while True:
            if len(index.uses) != 1:
                # If the induction variable is used more than once, we can't fold its
                # arith ops into the loop range
                return

            user = next(iter(index.uses)).operation

            if not isinstance(user, riscv.AddOp | riscv.MulOp):
                return

            if user.rs1 is index:
                if (imm := get_constant_value(user.rs2)) is None:
                    return
            else:
                if (imm := get_constant_value(user.rs1)) is None:
                    return

            constant = imm.value.data

            match user:
                case riscv.AddOp():
                    # All the uses are multiplications by a constant, we can fold
                    rewriter.insert_op_before_matched_op(
                        [
                            shift := riscv.LiOp(constant),
                            new_lb := riscv.AddOp(
                                op.lb, shift, rd=riscv.IntRegisterType.unallocated()
                            ),
                            new_ub := riscv.AddOp(
                                op.ub, shift, rd=riscv.IntRegisterType.unallocated()
                            ),
                        ]
                    )
                case riscv.MulOp():
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

                    op.operands[2] = new_step.rd

            op.operands[0] = new_lb.rd
            op.operands[1] = new_ub.rd
            rewriter.replace_op(user, [], [index])


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
