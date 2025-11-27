from xdsl.backend.riscv.lowering.utils import cast_operands_to_regs
from xdsl.context import Context
from xdsl.dialects import printf, riscv_debug
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerPrintFormatOp(RewritePattern):
    """
    Rewrites printf.PrintFormatOp to riscv_debug.printf.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        rewriter.replace_op(
            op, riscv_debug.PrintfOp(op.format_str, cast_operands_to_regs(rewriter))
        )


class ConvertPrintFormatToRiscvDebugPass(ModulePass):
    name = "convert-print-format-to-riscv-debug"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerPrintFormatOp()).rewrite_module(op)
