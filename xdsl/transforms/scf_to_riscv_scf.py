from typing import Sequence

from xdsl.dialects import builtin, riscv, riscv_scf, scf
from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def cast_vals_to_int_reg(
    vals: Sequence[SSAValue], rewriter: PatternRewriter, before: Operation
) -> list[SSAValue]:
    unallocated_reg = riscv.IntRegisterType.unallocated()
    new_vals = []
    for val in vals:
        if not isinstance(val.type, riscv.IntRegisterType):
            rewriter.insert_op_before(
                new_val := builtin.UnrealizedConversionCastOp([val], [unallocated_reg]),
                before,
            )
            new_vals.append(new_val.results[0])
        else:
            new_vals.append(val)
    return new_vals


def cast_vals_back(
    vals: Sequence[SSAValue], rewriter: PatternRewriter, before: Operation
):
    unallocated_reg = riscv.IntRegisterType.unallocated()

    for val in vals:
        rewriter.insert_op_before(
            new_val := builtin.UnrealizedConversionCastOp([val], [val.type]),
            before,
        )
        val.type = unallocated_reg
        val.replace_by(new_val.results[0])
        new_val.operands[new_val.results[0].index] = val


class ScfForToRiscvFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        cast_vals = cast_vals_to_int_reg(op.operands, rewriter, op)

        body = op.detach_region(0)
        cast_vals_back(body.block.args, rewriter, body.block.first_op)

        yield_op = body.block.last_op
        new_vals = cast_vals_to_int_reg(yield_op.operands, rewriter, yield_op)
        rewriter.replace_op(yield_op, riscv_scf.YieldOp(*new_vals))

        for res in op.results:
            rewriter.insert_op_after_matched_op(
                builtin.UnrealizedConversionCastOp([res], [res.type])
            )

        rewriter.replace_matched_op(
            riscv_scf.ForOp(
                cast_vals[0], cast_vals[1], cast_vals[2], cast_vals[3:], body
            ),
        )


class ScfToRiscvPass(ModulePass):
    name = "scf-to-riscv"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ScfForToRiscvFor()])
        ).rewrite_module(op)
