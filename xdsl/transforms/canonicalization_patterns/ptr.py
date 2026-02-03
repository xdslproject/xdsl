from xdsl.dialects import arith, ptr
from xdsl.dialects.builtin import IntegerAttr
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
            rewriter.replace_op(op, (), (origin.source,))


class RedundantToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.ToPtrOp, rewriter: PatternRewriter, /):
        if not isinstance(origin := op.source.owner, ptr.FromPtrOp):
            return

        if origin.source.type == op.res.type:
            rewriter.replace_op(op, (), (origin.source,))


class PtrAddZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter, /):
        if (
            (isinstance(offset_op := op.offset.owner, arith.ConstantOp))
            and isinstance(offset_op.value, IntegerAttr)
            and offset_op.value.value.data == 0
        ):
            rewriter.replace_op(op, (), (op.addr,))
