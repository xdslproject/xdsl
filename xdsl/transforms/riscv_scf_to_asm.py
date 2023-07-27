from xdsl.dialects import builtin, riscv, riscv_scf
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerRiscvScfToLabels(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp, rewriter: PatternRewriter, /):
        # get loop var
        loop_var = op.lb
        # insert leader:
        # TODO: only works if loop variable register == lb register
        rewriter.insert_op_before_matched_op(
            [
                riscv.LabelOp("scf_leader"),
                riscv.BgeOp(loop_var, op.ub, riscv.LabelAttr("scf_body_end")),
                # start of loop body:
                riscv.LabelOp("scf_body"),
            ]
        )
        # add latch to body
        body = op.body.detach_block(0)
        yield_op = body.last_op
        assert isinstance(yield_op, riscv_scf.YieldOp)
        body.insert_ops_after(
            [
                # increment loop var in-place
                riscv.AddOp(loop_var, op.step, rd=loop_var.type),
                # branch to start of loop
                riscv.BltOp(loop_var, op.ub, riscv.LabelAttr("scf_body")),
                # add end of loop thingy
                riscv.LabelOp("scf_body_end"),
            ],
            yield_op,
        )
        body.erase_op(yield_op)
        for arg in body.args:
            val = riscv.GetRegisterOp(arg.type)
            body.insert_op_before(val, body.first_op)
            arg.replace_by(val.res)

        # insert body with latch:
        rewriter.inline_block_after_matched_op(body)
        for loop_result in op.results:
            # rewrite result with var
            val = riscv.GetRegisterOp(loop_result.type)
            loop_result.replace_by(val.res)
            rewriter.insert_op_after_matched_op(val)
        rewriter.erase_matched_op()


class LowerScfForToLabels(ModulePass):
    name = "lower-riscv-scf-to-labels"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerRiscvScfToLabels()).rewrite_module(op)
