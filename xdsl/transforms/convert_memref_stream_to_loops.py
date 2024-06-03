from collections.abc import Sequence

from xdsl.dialects import memref, memref_stream, stream
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.ir import MLContext, Operation, SSAValue
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
    source: SSAValue,
    indices: Sequence[SSAValue],
    rewriter: PatternRewriter,
    insert_point: InsertPoint,
) -> SSAValue:
    if isinstance(source.type, memref.MemRefType):
        op = memref.Load.get(source, indices)
    elif isinstance(source.type, stream.ReadableStreamType):
        op = memref_stream.ReadOp(source)
    else:
        return source
    rewriter.insert_op_at_location(op, insert_point)
    return op.res


def store(
    value: SSAValue, destination: SSAValue, indices: Sequence[SSAValue]
) -> Operation:
    if isinstance(destination.type, memref.MemRefType):
        return memref.Store.get(value, destination, indices)
    else:
        return memref_stream.WriteOp(value, destination)


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        rewrite_generic_to_loops(rewriter, op, load, store)


class ConvertMemrefStreamToLoopsPass(ModulePass):
    """
    Converts a memref_stream generic to loop.
    """

    name = "convert-memref-stream-to-loops"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
