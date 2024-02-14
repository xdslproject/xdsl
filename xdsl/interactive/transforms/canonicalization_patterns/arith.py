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

arith_operations_that_have_canonicalization_patterns: list[type[Operation]] = [
    arith.Addi
]
"""
Contains the list of arith operations that are the root of a defined rewrite pattern.
"""


def get_interactive_arith_rewrite_patterns() -> tuple[RewritePattern, ...]:
    """
    Returns the list of experimental arith rewrite patterns.
    """
    return (AdditionOfSameVariablesToMultiplyByTwo(),)


def operation_has_interactive_rewrite_pattern(op: type[Operation]) -> bool:
    """
    Function that checks if an operation has (one) or many interactive rewrite pattern.
    """
    for op_type in arith_operations_that_have_canonicalization_patterns:
        if op == op_type:
            return True
    return False


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
