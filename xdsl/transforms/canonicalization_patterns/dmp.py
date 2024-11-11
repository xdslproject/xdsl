from xdsl.dialects import stencil
from xdsl.dialects.experimental import dmp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class DmpSwapNoExchangeFolding(RewritePattern):
    """
    Folds `dmp.swap` ops that contain no swaps. Needs to be run after shape inference and before bufferization.
    Checks the input and result of the matched op to see if that appears to be the case.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        if (
            op.swapped_values
            and isinstance(temp_t := op.input_stencil.type, stencil.TempType)
            and isinstance(temp_t.bounds, stencil.StencilBoundsAttr)
            and isinstance(res_t := op.swapped_values.type, stencil.TempType)
            and isinstance(res_t.bounds, stencil.StencilBoundsAttr)
            and not op.swaps
        ):
            rewriter.replace_matched_op([], new_results=[op.input_stencil])
