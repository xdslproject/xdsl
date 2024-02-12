from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.lhs.owner, arith.Constant)
            and isinstance(value := op.lhs.owner.value, IntegerAttr)
            and value.value.data == 0
        ):
            rewriter.replace_matched_op([], [op.rhs])
