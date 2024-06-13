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
from xdsl.transforms.loop_nest_lowering_utils import (
    rewrite_generic_to_imperfect_loops,
    rewrite_generic_to_loops,
)


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
    rewriter.insert_op(op, insert_point)
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
        ubs = op.get_static_loop_ranges()
        outer_bound_count = min(len(ubs), *(m.data.num_dims for m in op.indexing_maps))
        if outer_bound_count != len(ubs):
            # Imperfectly nested
            ins_count = len(op.inputs)
            rewrite_generic_to_imperfect_loops(
                rewriter,
                InsertPoint.before(op),
                ubs[:outer_bound_count],
                ubs[outer_bound_count:],
                op.indexing_maps.data[ins_count:],
                op.indexing_maps.data[:ins_count],
                op.indexing_maps.data[ins_count:],
                op.outputs,
                op.inputs,
                op.outputs,
                op.body.block.args[ins_count:],
                op.body.block.args[:ins_count],
                op.body.block,
                load,
                store,
            )
        else:
            rewrite_generic_to_loops(
                rewriter,
                InsertPoint.before(op),
                ubs,
                op.indexing_maps.data,
                op.indexing_maps.data[-len(op.outputs) :],
                op.operands,
                op.outputs,
                op.body.block,
                load,
                store,
            )


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
