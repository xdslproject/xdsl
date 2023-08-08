from typing import Iterator, Sequence

from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

_REG_TYPES = riscv.IntRegisterType | riscv.FloatRegisterType


def replace_values_by_registers_op(
    values: Sequence[SSAValue],
) -> Iterator[riscv.GetRegisterOp | riscv.GetFloatRegisterOp]:
    for value in values:
        assert isinstance(value.type, _REG_TYPES)

        get_target_register = (
            riscv.GetRegisterOp(value.type)
            if isinstance(value.type, riscv.IntRegisterType)
            else riscv.GetFloatRegisterOp(value.type)
        )

        value.replace_by(get_target_register.res)
        yield get_target_register


class LowerRiscvScfToLabels(RewritePattern):
    for_idx = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter, /):
        # The first argument of the loop body block is the loop counter by SCF invariant.
        loop_var_reg = op.body.block.args[0].type
        assert isinstance(loop_var_reg, riscv.IntRegisterType)

        # To ensure that we have a unique labels for each (nested) loop, we use an
        # index that is incremented for each loop as a suffix.
        suffix = f"{self.for_idx}_for"
        scf_cond = riscv.LabelAttr(f"scf_cond_{suffix}")
        scf_body = riscv.LabelAttr(f"scf_body_{suffix}")
        scf_body_end = riscv.LabelAttr(f"scf_body_end_{suffix}")

        # This is the loop header, responsible for comparing the loop counter to the
        # upper bound and branching to the loop body if the condition is met.
        rewriter.insert_op_before_matched_op(
            [
                get_loop_var := riscv.GetRegisterOp(loop_var_reg),
                riscv.LabelOp(scf_cond),
                riscv.BgeOp(get_loop_var, op.ub, scf_body_end),
                riscv.LabelOp(scf_body),
            ]
        )

        # Append the loop increment and the branch back to the loop header
        # to the end of the body.

        body = op.body.detach_block(0)
        yield_op = body.last_op
        assert isinstance(yield_op, riscv_scf.YieldOp)

        body.insert_ops_after(
            [
                riscv.AddOp(get_loop_var, op.step, rd=loop_var_reg),
                riscv.BltOp(get_loop_var, op.ub, scf_body),
                riscv.LabelOp(scf_body_end),
            ],
            yield_op,
        )
        body.erase_op(yield_op)

        # We know that the body is not empty now.
        assert body.first_op is not None

        # Replace args of the body with operations that get the registers bound
        # to them.
        for get_target_register in replace_values_by_registers_op(body.args):
            body.insert_op_before(get_target_register, body.first_op)

        # Also replace the loop results directly with the registers bound to them.
        for get_target_register in replace_values_by_registers_op(op.results):
            rewriter.insert_op_after_matched_op(get_target_register)

        # Extract ops from the body and insert them after the loop header.
        rewriter.inline_block_after_matched_op(body)

        rewriter.erase_matched_op()

        self.for_idx += 1


class LowerScfForToLabels(ModulePass):
    name = "lower-riscv-scf-to-labels"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerRiscvScfToLabels()).rewrite_module(op)
