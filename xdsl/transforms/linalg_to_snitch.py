from abc import ABC
from dataclasses import dataclass

from xdsl.dialects import linalg, snitch
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddSnitchStreamEnableDisable(RewritePattern, ABC):
    """
    Adds Snitch stream enable and disable instructions.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        rewriter.insert_op_before_matched_op(snitch.SsrEnable())
        rewriter.insert_op_after_matched_op(snitch.SsrDisable())


@dataclass
class LowerLinalgToSnitchPass(ModulePass):
    """ """

    name = "lower-linalg-to-snitch"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddSnitchStreamEnableDisable(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
