from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class AddImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.lhs.owner, arith.Constant)
            and isinstance(value := op.lhs.owner.value, IntegerAttr)
            and value.value.data == 0
        ):
            rewriter.replace_matched_op([], [op.rhs])


class AdditionOfSameVariablesToMultiplyByTwo(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if op.lhs == op.rhs:
            rd = arith.cast(arith.IntegerType, op.lhs.type)
            assert isa(op.lhs.type, IntegerType | IndexType)
            rewriter.replace_matched_op(
                [
                    li_op := arith.Constant(IntegerAttr(2, op.lhs.type)),
                    arith.Muli(op.lhs, li_op, rd),
                ]
            )
