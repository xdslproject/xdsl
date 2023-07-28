from dataclasses import dataclass, field

from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class LowerRiscvScfToLabels(RewritePattern):
    cnt: int = field(default=-1)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter, /):
        self.cnt += 1

        # get loop var
        loop_var = op.lb
        # insert leader:
        # TODO: only works if loop variable register == lb register
        rewriter.insert_op_before_matched_op(
            [
                riscv.LabelOp(f"scf_leader_{self.cnt}"),
                riscv.BgeOp(
                    loop_var, op.ub, riscv.LabelAttr(f"scf_body_end_{self.cnt}")
                ),
                # start of loop body:
                riscv.LabelOp(f"scf_body_{self.cnt}"),
            ]
        )
        # add latch to body
        body = op.body.detach_block(0)
        yield_op = body.last_op
        assert isinstance(yield_op, riscv_scf.YieldOp)
        assert isinstance(loop_var.type, riscv.IntRegisterType)
        body.insert_ops_after(
            [
                # increment loop var in-place
                riscv.AddOp(loop_var, op.step, rd=loop_var.type),
                # branch to start of loop
                riscv.BltOp(loop_var, op.ub, riscv.LabelAttr(f"scf_body_{self.cnt}")),
                # add end of loop thingy
                riscv.LabelOp(f"scf_body_end_{self.cnt}"),
            ],
            yield_op,
        )
        body.erase_op(yield_op)
        for arg in body.args:
            assert isinstance(arg.type, riscv.IntRegisterType)
            val = riscv.GetRegisterOp(arg.type)
            rewriter.insert_op_at_start(val, body)
            arg.replace_by(val.res)

        # insert body with latch:
        rewriter.inline_block_after_matched_op(body)
        for loop_result in op.results:
            assert isinstance(loop_result.type, riscv.IntRegisterType)
            # rewrite result with var
            val = riscv.GetRegisterOp(loop_result.type)
            loop_result.replace_by(val.res)
            rewriter.insert_op_after_matched_op(val)
        rewriter.erase_matched_op()


class LowerScfForToLabelsPass(ModulePass):
    name = "lower-riscv-scf-to-labels"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerRiscvScfToLabels()).rewrite_module(op)
