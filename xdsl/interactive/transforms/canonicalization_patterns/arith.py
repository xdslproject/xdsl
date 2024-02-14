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


arith_op_to_rewrite_pattern: dict[type[Operation], tuple[RewritePattern, ...]] = {
    arith.Addi: tuple((AdditionOfSameVariablesToMultiplyByTwo(),))
}
"""
Dictionary where the key is an Operation and the value is a tuple of rewrite pattern(s) associated with that operation.
"""
