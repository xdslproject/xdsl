from xdsl.ir import MLContext
from xdsl.pattern_rewriter import (
    PatternRewriter,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
)
from xdsl.dialects import arith, riscv, builtin, scf
from xdsl.passes import ModulePass


class LowerArithConstantOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter, /):
        assert isinstance(op.value, builtin.IntegerAttr)

        rewriter.replace_matched_op(
            riscv.LiOp(op.value, rd=riscv.RegisterType(riscv.Register()))
        )


class LowerArithBinaryOpToRiscv(RewritePattern):
    ops: dict[str, type[riscv.RdRsRsOperation]] = {"arith.addi": riscv.AddOp}

    def select_instruction(self, op: arith.BinaryOperation) -> riscv.RdRsRsOperation:
        if op.name not in self.ops:
            raise ValueError(f"Cannot lower {op.name}")

        dest = self.ops.get(op.name)
        return dest(*op.operands, rd=riscv.RegisterType(riscv.Register()))

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.BinaryOperation, rewriter: PatternRewriter, /
    ):
        new_op = self.select_instruction(op)
        rewriter.replace_matched_op(new_op)


class LowerScfForToRiscv(RewritePattern):
    label_cnt: int

    def __init__(self):
        self.label_cnt = 0

    def _next_label(self) -> int:
        self.label_cnt += 1
        return self.label_cnt

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        # grab a unique start and end label for the loop
        start_lbl = self._next_label()
        end_lbl = self._next_label()

        # sentinel-allocate loop register
        loop_register = riscv.Register(f"loop-var-{self._next_label()}")

        # insert the loop guard and start label
        rewriter.insert_op_before_matched_op(
            [
                riscv.BgeOp(op.lb, op.ub, riscv.LabelAttr(end_lbl, "f")),
                loop_var := riscv.AddiOp(
                    op.lb, 0, rd=loop_register
                ),  # TODO: make this a move op!
                riscv.LabelOp(start_lbl),
            ]
        )

        # replace the loop variable in the for body with our loop register
        assert len(op.body.block.args) == 1
        op.body.block.args[0].replace_by(loop_var.rd)
        # insert loop body into top-level
        rewriter.inline_block_before_matched_op(op.body.block)

        # add loop counter and jump back
        rewriter.replace_matched_op(
            [
                riscv.AddOp(loop_var, op.step, rd=loop_register),
                riscv.BltOp(loop_var, op.ub, riscv.LabelAttr(start_lbl, "b")),
                riscv.LabelOp(end_lbl),
            ],
            [],
        )


class RiscvLoweringPass(ModulePass):
    name = "lower-riscv"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerArithConstantOp(),
                    LowerArithBinaryOpToRiscv(),
                    LowerScfForToRiscv(),
                ]
            ),
            apply_recursively=True,
        ).rewrite_module(op)
