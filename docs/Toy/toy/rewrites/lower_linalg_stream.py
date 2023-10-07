"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""

from xdsl.dialects import linalg, stream
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerGenericOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter):
        if op.res:
            raise NotImplementedError("Cannot lower linalg generic op with results")
        rewriter.replace_matched_op(
            stream.GenericOp(
                (),
                op.inputs,
                (),
                op.outputs,
                rewriter.move_region_contents_to_new_regions(op.body),
                op.indexing_maps,
                None,
            )
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Yield, rewriter: PatternRewriter):
        rewriter.replace_matched_op(stream.YieldOp(*op.operands))


class LinalgToStreamPass(ModulePass):
    name = "linalg-to-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
