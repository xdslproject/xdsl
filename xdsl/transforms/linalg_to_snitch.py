from abc import ABC
from dataclasses import dataclass

from xdsl.dialects import linalg, riscv, snitch
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32
from xdsl.ir import MLContext  # noqa: E999
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class AddSnitchStreamLoopBoundAndStrideConfig(RewritePattern, ABC):
    """
    Adds Snitch stream configuration instructions for the loop bound and stride at a
    specific loop depth.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        for loop_idx in range(0, op.get_num_loops()):
            stream = riscv.LiOp(31)
            bound = riscv.LiOp(364)
            dim = IntegerAttr(loop_idx, i32)
            ssr_bound_op = snitch.SsrSetDimensionBoundOp(stream, bound, dim)
            rewriter.insert_op_before_matched_op([stream, bound, ssr_bound_op])


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
        walker1 = PatternRewriteWalker(
            AddSnitchStreamLoopBoundAndStrideConfig(), apply_recursively=False
        )
        walker2 = PatternRewriteWalker(
            AddSnitchStreamEnableDisable(), apply_recursively=False
        )
        walker1.rewrite_module(op)
        walker2.rewrite_module(op)
