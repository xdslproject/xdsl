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


class SubSquaresPattern(RewritePattern):
    """
    a²-b²=(a-b)(a+b)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subi, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.lhs.owner, arith.Muli)
            and (a := op.lhs.owner.lhs) is op.lhs.owner.rhs
            and isinstance(op.rhs.owner, arith.Muli)
            and (b := op.rhs.owner.lhs) is op.rhs.owner.rhs
        ):
            rewriter.replace_matched_op(
                (
                    diff := arith.Subi(a, b),
                    sum := arith.Addi(a, b),
                    arith.Muli(diff.result, sum.result),
                )
            )
