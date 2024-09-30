from xdsl.dialects import arith, cf
from xdsl.dialects.builtin import IntegerAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AssertTrue(RewritePattern):
    """Erase assertion if argument is constant true."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cf.Assert, rewriter: PatternRewriter):
        owner = op.arg.owner

        if not isinstance(owner, arith.Constant):
            return

        value = owner.value

        if not isinstance(value, IntegerAttr):
            return

        if value.value.data != 1:
            return

        rewriter.replace_matched_op([])
