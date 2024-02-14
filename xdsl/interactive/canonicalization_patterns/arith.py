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


def get_interactive_arith_canonicalization_patterns() -> tuple[RewritePattern, ...]:
    return (AdditionOfSameVariablesToMultiplyByTwo(),)


def has_interactive_canonicalization_pattern(op: type[Operation]) -> bool:
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
