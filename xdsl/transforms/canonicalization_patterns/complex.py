from xdsl.dialects import complex
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class FoldConstConstOp(RewritePattern):
    """
    Folds a complex floating point binary op whose operands are both `complex.constant`s.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: complex.ComplexBinaryOp,
        rewriter: PatternRewriter,
    ):
        if (attr := op.fold()) is not None:
            rewriter.replace_matched_op(
                complex.ConstantOp(
                    ArrayAttr(
                        [
                            FloatAttr(attr[0].real, attr[0].type.element_type),
                            FloatAttr(attr[0].imag, attr[0].type.element_type),
                        ]
                    ),
                    op.result.type,
                )
            )
