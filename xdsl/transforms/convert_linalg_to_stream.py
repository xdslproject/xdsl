"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""

from xdsl.dialects import linalg, stream
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
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
            # Cannot lower linalg generic op with results
            return

        ub = op.get_static_loop_ranges()
        ub_attr = ArrayAttr(IntAttr(b) for b in ub)

        new_inputs = [
            stream.StridedReadOp(memref, ub_attr, indexing_map)
            for memref, indexing_map in zip(op.inputs, op.indexing_maps)
        ]
        new_outputs = [
            stream.StridedWriteOp(memref, ub_attr, indexing_map)
            for memref, indexing_map in zip(
                op.outputs, op.indexing_maps.data[-len(op.outputs) :]
            )
        ]

        rewriter.replace_matched_op(
            [
                *new_inputs,
                *new_outputs,
                stream.GenericOp(
                    [i.stream for i in new_inputs],
                    [o.stream for o in new_outputs],
                    rewriter.move_region_contents_to_new_regions(op.body),
                    ub_attr,
                ),
            ]
        )


class LowerYieldOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Yield, rewriter: PatternRewriter):
        rewriter.replace_matched_op(stream.YieldOp(*op.operands))


class ConvertLinalgToStreamPass(ModulePass):
    name = "convert-linalg-to-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    LowerYieldOp(),
                ]
            )
        ).rewrite_module(op)
