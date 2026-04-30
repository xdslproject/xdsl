"""
This pass converts a statically-known `step` value into a constant attribute in
`riscv_scf` loops when it fits into the bitwidth of the immediate in the `addi`
operation.
The rationale is to use one fewer register in the loop body.
The lower bound and upper bound don't benefit from the same transformation, as the
former is only used to initialise the induction variable, and the step is used in a
comparison which always takes two registers.
"""

from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_scf, rv32
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.riscv.attrs import si12
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import VerifyException


class InferConstantStep(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_scf.ForOp | riscv_scf.RofOp, rewriter: PatternRewriter
    ) -> None:
        if (sv := op.step_val) is None:
            # Nothing to be done
            return

        if not isinstance(owner := sv.owner, rv32.LiOp):
            # Only support rv32.LiOp for now
            return

        if isinstance(imm := owner.immediate, riscv.LabelAttr):
            # Cannot support labels for now
            return

        try:
            step_attr = IntegerAttr(imm.value.data, si12)
        except VerifyException:
            # RV32 I-type addi immediates (12-bit signed). This is only valid until we
            # add support for rv64, revisit then.
            return

        new_op = type(op)(
            op.lb,
            op.ub,
            step_attr,
            op.iter_args,
            rewriter.move_region_contents_to_new_regions(op.body),
        )
        rewriter.replace_op(op, new_op, new_op.results)


class RiscvScfForInferConstantStepPass(ModulePass):
    name = "riscv-scf-for-infer-constant-step"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(InferConstantStep()).rewrite_module(op)
