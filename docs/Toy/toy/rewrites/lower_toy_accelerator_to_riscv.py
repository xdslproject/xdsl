from xdsl.backend.riscv.lowering.utils import cast_operands_to_regs
from xdsl.dialects import memref, riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..dialects import toy_accelerator


class LowerTransposeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: toy_accelerator.Transpose, rewriter: PatternRewriter
    ):
        destination, source = cast_operands_to_regs(rewriter)

        rewriter.replace_matched_op(
            [
                rows_op := riscv.LiOp(op.source_rows.value.data, comment="source rows"),
                cols_op := riscv.LiOp(op.source_cols.value.data, comment="source cols"),
                riscv.CustomAssemblyInstructionOp(
                    "tensor.transpose2d",
                    (destination, source, rows_op.rd, cols_op.rd),
                    (),
                ),
            ]
        )


class LowerBinOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy_accelerator.BinOp, rewriter: PatternRewriter):
        typ = op.dest.type
        assert isinstance(typ, memref.MemRefType)
        size = typ.element_count()

        instruction_name = (
            "buffer.add" if isinstance(op, toy_accelerator.Add) else "buffer.mul"
        )
        dest, lhs, rhs = cast_operands_to_regs(rewriter)
        rewriter.replace_matched_op(
            [
                size_op := riscv.LiOp(size, comment="size"),
                riscv.CustomAssemblyInstructionOp(
                    "buffer.copy",
                    (size_op.rd, dest, lhs),
                    (),
                ),
                riscv.CustomAssemblyInstructionOp(
                    instruction_name,
                    (size_op.rd, dest, rhs),
                    (),
                ),
            ]
        )


class LowerToyAccelerator(ModulePass):
    name = "lower-toy-accelerator"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTransposeOp(),
                    LowerBinOp(),
                ]
            )
        ).rewrite_module(op)
