from xdsl.dialects.csl import csl
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class GetMemDsdAndStrideOpFolding(RewritePattern):
    """
    Removes redundant allocations of empty tensors with no uses other than passed
    as `iter_arg` to `csl_stencil.apply`. Prefer re-use where possible.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        pass
