from xdsl.dialects import ptr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RedundantFromPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.FromPtrOp, rewriter: PatternRewriter, /):
        if not isinstance(origin := op.source.owner, ptr.ToPtrOp):
            return

        if origin.source.type == op.res.type:
            rewriter.replace_matched_op((), (origin.source,))


class RedundantToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.ToPtrOp, rewriter: PatternRewriter, /):
        if not isinstance(origin := op.source.owner, ptr.FromPtrOp):
            return

        if origin.source.type == op.res.type:
            rewriter.replace_matched_op((), (origin.source,))
