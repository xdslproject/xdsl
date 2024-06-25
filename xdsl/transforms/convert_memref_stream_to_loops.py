from collections.abc import Sequence

from xdsl.context import MLContext
from xdsl.dialects import memref, memref_stream, stream
from xdsl.dialects.builtin import AffineMapAttr, ModuleOp
from xdsl.ir import Operation, SSAValue
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
    indices_for_map,
    rewrite_generic_to_imperfect_loops,
    rewrite_generic_to_loops,
)


def insert_load(
    source: SSAValue,
    affine_map_attr: AffineMapAttr,
    ind_vars: Sequence[SSAValue],
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
) -> SSAValue:
    if isinstance(source.type, memref.MemRefType):
        indices = indices_for_map(
            rewriter, insertion_point, affine_map_attr.data, ind_vars
        )
        op = memref.Load.get(source, indices)
    elif isinstance(source.type, stream.ReadableStreamType):
        op = memref_stream.ReadOp(source)
    else:
        return source
    rewriter.insert_op(op, insertion_point)
    return op.res


def insert_store(
    value: SSAValue,
    destination: SSAValue,
    affine_map_attr: AffineMapAttr,
    ind_vars: Sequence[SSAValue],
    rewriter: PatternRewriter,
    insertion_point: InsertPoint,
) -> Operation:
    if isinstance(destination.type, memref.MemRefType):
        indices = indices_for_map(
            rewriter, insertion_point, affine_map_attr.data, ind_vars
        )
        op = memref.Store.get(value, destination, indices)
    else:
        op = memref_stream.WriteOp(value, destination)
    rewriter.insert_op(op, insertion_point)
    return op


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        outer_ubs, inner_ubs = op.get_static_loop_ranges()
        if inner_ubs:
            # Imperfectly nested
            ins_count = len(op.inputs)
            rewrite_generic_to_imperfect_loops(
                rewriter,
                InsertPoint.before(op),
                outer_ubs,
                inner_ubs,
                op.indexing_maps.data[ins_count:],
                op.indexing_maps.data[:ins_count],
                op.indexing_maps.data[ins_count:],
                op.outputs,
                op.inputs,
                op.outputs,
                op.body.block.args[ins_count:],
                op.body.block.args[:ins_count],
                op.body.block,
                insert_load,
                insert_store,
            )
        else:
            rewrite_generic_to_loops(
                rewriter,
                InsertPoint.before(op),
                outer_ubs,
                op.indexing_maps.data,
                op.indexing_maps.data[-len(op.outputs) :],
                op.operands,
                op.outputs,
                op.body.block,
                insert_load,
                insert_store,
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
