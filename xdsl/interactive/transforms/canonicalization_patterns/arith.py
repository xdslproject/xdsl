"""
Since we might not want to add "bad" arith rewrite patterns to be used outside of
xdsl-gui, in this file, we define arith rewrite patterns that can only be used by xdsl-gui.
"""


from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AdditionOfSameVariablesToMultiplyByTwo(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if op.lhs == op.rhs:
            assert isinstance(op.lhs.type, IntegerType | IndexType)
            rewriter.replace_matched_op(
                [
                    li_op := arith.Constant(IntegerAttr(2, op.lhs.type)),
                    arith.Muli(op.lhs, li_op),
                ]
            )


class DivisionOfSameVariableToOne(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUI, rewriter: PatternRewriter) -> None:
        if (
            isinstance(mul_op := op.lhs.owner, Operation)
            and isinstance(mul_op, arith.Muli)
            and (op.lhs in mul_op.results)
            # and mul_op.rhs == op.rhs
            and isinstance(mul_op.rhs.owner, arith.Constant)
            and isinstance(mul_rhs_value := mul_op.rhs.owner.value, IntegerAttr)
            and isinstance(op.rhs.owner, arith.Constant)
            and isinstance(value := op.rhs.owner.value, IntegerAttr)
            and mul_rhs_value.value.data == value.value.data
            and value.value.data != 0
        ):
            rewriter.replace_matched_op([], [mul_op.lhs])
            assert isinstance(mul_op.rhs.owner, Operation)
            try:
                rewriter.erase_op(mul_op)
                rewriter.erase_op(mul_op.rhs.owner)
            except Exception as e:
                # would not allow bare except, whats the g
                print(e)

            try:
                rewriter.erase_op(op.rhs.owner)
            except Exception as e:
                print(e)


arith_op_to_rewrite_pattern: dict[type[Operation], tuple[RewritePattern, ...]] = {
    arith.Addi: tuple((AdditionOfSameVariablesToMultiplyByTwo(),)),
    arith.DivUI: tuple((DivisionOfSameVariableToOne(),)),
}
"""
Dictionary where the key is an Operation and the value is a tuple of rewrite pattern(s) associated with that operation.
"""
