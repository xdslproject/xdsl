from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)

from xdsl.dialects import arith


class ConstantFolding(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.op, arith.ConstantOp) and isinstance(
            op.rhs.op, arith.ConstantOp
        ):  # pattern: if both arguments to the Addi operation are from `Constant` operations
            rewriter.replace_matched_op(  # transform: replace the operation by calculating the sum of the constants at compile time
                arith.ConstantOp.from_int_and_width(
                    op.lhs.op.value.value.data + op.rhs.op.value.value.data,
                    op.lhs.op.value.type.width.data,
                )
            )
