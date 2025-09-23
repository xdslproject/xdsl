from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import transform
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class EraseTransformNamedSequenceOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transform.NamedSequenceOp, rewriter: PatternRewriter
    ) -> None:
        rewriter.erase_op(op)


@dataclass(frozen=True)
class TestTransformDialectEraseSchedulePass(ModulePass):
    """
    Erases transform named sequence operations.
    """

    name = "test-transform-dialect-erase-schedule"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            EraseTransformNamedSequenceOps(),
            apply_recursively=False,
        ).rewrite_module(op)
