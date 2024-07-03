from collections.abc import Sequence

from xdsl.context import MLContext
from xdsl.dialects import memref, memref_stream, stream
from xdsl.dialects.builtin import AffineMapAttr, ModuleOp, UnitAttr
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
from xdsl.utils.exceptions import DiagnosticException


def _insert_load(
    source_index: int,
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
        if memref_stream.IteratorTypeAttr.interleaved() in op.iterator_types:
            raise DiagnosticException("Cannot yet lower interleaved iterators")
        ins_count = len(op.inputs)
        if any(not isinstance(init, UnitAttr) for init in op.inits):
            constant_vals: list[SSAValue | None] = [None] * len(op.outputs)
            for index, val in zip(op.init_indices, op.inits, strict=True):
                constant_vals[index.data] = val

            def insert_load(
                source_index: int,
                source: SSAValue,
                affine_map_attr: AffineMapAttr,
                ind_vars: Sequence[SSAValue],
                rewriter: PatternRewriter,
                insertion_point: InsertPoint,
            ) -> SSAValue:
                if source_index >= ins_count:
                    constant_val = constant_vals[source_index - ins_count]
                    if constant_val is not None:
                        return constant_val

                return _insert_load(
                    source_index,
                    source,
                    affine_map_attr,
                    ind_vars,
                    rewriter,
                    insertion_point,
                )

        else:
            insert_load = _insert_load

        outer_ubs, inner_ubs = op.get_static_loop_ranges()
        if inner_ubs:
            # Imperfectly nested
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
