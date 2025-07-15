from xdsl.dialects import x86
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveRedundantDS_Mov(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: x86.DS_MovOp, rewriter: PatternRewriter) -> None:
        if op.destination.type == op.source.type and op.destination.type.is_allocated:
            rewriter.replace_matched_op((), (op.source,))
