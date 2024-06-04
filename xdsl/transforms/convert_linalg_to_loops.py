from collections.abc import Sequence

from xdsl.dialects import linalg, memref
from xdsl.dialects.builtin import MemRefType, ModuleOp
from xdsl.ir import MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.loop_nest_lowering_utils import rewrite_generic_to_loops


def load(
    value: SSAValue,
    indices: Sequence[SSAValue],
    rewriter: PatternRewriter,
    insertion_target: InsertPoint,
) -> SSAValue:
    if isinstance(value.type, MemRefType):
        op = memref.Load.get(value, indices)
        rewriter.insert_op(op, insertion_target)
        return op.res
    else:
        return value


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter) -> None:
        if op.res:
            raise NotImplementedError(
                "lowering for linalg.generic with results not yet supported"
            )

        rewrite_generic_to_loops(
            rewriter,
            InsertPoint.before(op),
            op.get_static_loop_ranges(),
            op.indexing_maps.data,
            op.indexing_maps.data[-len(op.outputs) :],
            op.operands,
            op.outputs,
            op.body.block,
            load,
            memref.Store.get,
        )


class ConvertLinalgToLoopsPass(ModulePass):
    """
    Converts a linalg generic to perfectly nested loops.
    """

    name = "convert-linalg-to-loops"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
