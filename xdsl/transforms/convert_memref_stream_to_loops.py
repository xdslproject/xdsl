from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import arith, memref, memref_stream
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, ModuleOp, UnitAttr
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
        op = memref.LoadOp.get(source, indices)
    elif isinstance(source.type, memref_stream.ReadableStreamType):
        op = memref_stream.ReadOp(source)
    else:
        return source
    rewriter.insert_op(op, insertion_point)
    return op.res


class LowerGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if memref_stream.IteratorTypeAttr.interleaved() in op.iterator_types:
            interleave_factor = op.bounds.data[-1].value.data
            rewriter.insert_op(
                interleaved_index_ops := tuple(
                    arith.ConstantOp(IntegerAttr.from_index_int_value(i))
                    for i in range(interleave_factor)
                )
            )
            interleaved_index_vals = tuple(op.result for op in interleaved_index_ops)

            def extra_dim(source_index: int) -> tuple[SSAValue] | tuple[()]:
                return (interleaved_index_vals[source_index % interleave_factor],)

        else:
            interleave_factor = 1
            interleaved_index_vals = ()

            def extra_dim(source_index: int) -> tuple[SSAValue] | tuple[()]:
                return ()

        ins_count = len(op.inputs) * interleave_factor

        if any(not isinstance(init, UnitAttr) for init in op.inits):
            constant_vals: list[SSAValue | None] = [None] * len(op.outputs)
            for index, val in zip(op.init_indices, op.inits, strict=True):
                constant_vals[index.data] = val

            def insert_load(
                source_index: int,
                ind_vars: Sequence[SSAValue],
                rewriter: PatternRewriter,
                insertion_point: InsertPoint,
            ) -> SSAValue:
                """
                Inserts a load op or returns an initial value for a given operand of the
                generic operation being lowered.
                If the operation contains an INTERLEAVED iterator type then the body has a
                corresponding factor of duplication of arguments.
                To determine the appropriate operand and indexing map, the source_index is
                divided by the interleave factor.
                """
                source = op.operands[source_index // interleave_factor]
                affine_map_attr = op.indexing_maps.data[
                    source_index // interleave_factor
                ]
                if source_index >= ins_count:
                    constant_val = constant_vals[
                        (source_index - ins_count) // interleave_factor
                    ]
                    if constant_val is not None:
                        return constant_val

                return _insert_load(
                    source_index,
                    source,
                    affine_map_attr,
                    tuple(ind_vars) + extra_dim(source_index),
                    rewriter,
                    insertion_point,
                )

        else:

            def insert_load(
                source_index: int,
                ind_vars: Sequence[SSAValue],
                rewriter: PatternRewriter,
                insertion_point: InsertPoint,
            ) -> SSAValue:
                """
                Inserts a load op for a given operand of the generic operation being
                lowered.
                If the operation contains an INTERLEAVED iterator type then the body has a
                corresponding factor of duplication of arguments.
                To determine the appropriate operand and indexing map, the source_index is
                divided by the interleave factor.
                """
                source = op.operands[source_index // interleave_factor]
                affine_map_attr = op.indexing_maps.data[
                    source_index // interleave_factor
                ]
                return _insert_load(
                    source_index,
                    source,
                    affine_map_attr,
                    tuple(ind_vars) + extra_dim(source_index),
                    rewriter,
                    insertion_point,
                )

        def insert_store(
            source_index: int,
            value: SSAValue,
            ind_vars: Sequence[SSAValue],
            rewriter: PatternRewriter,
            insertion_point: InsertPoint,
        ) -> Operation:
            """
            Inserts a store op for a given operand of the generic operation being lowered.
            If the operation contains an INTERLEAVED iterator
            type then the body has a corresponding factor of duplication of arguments.
            To determine the appropriate operand and indexing map, the source_index is
            divided by the interleave factor.
            """
            nonlocal op
            index = (source_index + ins_count) // interleave_factor
            destination = op.operands[index]
            affine_map = op.indexing_maps.data[index].data
            if isinstance(destination.type, memref.MemRefType):
                indices = indices_for_map(
                    rewriter,
                    insertion_point,
                    affine_map,
                    tuple(ind_vars) + extra_dim(source_index),
                )
                store_op = memref.StoreOp.get(value, destination, indices)
            else:
                store_op = memref_stream.WriteOp(value, destination)
            rewriter.insert_op(store_op, insertion_point)
            return store_op

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


class ConvertMemRefStreamToLoopsPass(ModulePass):
    """
    Converts a memref_stream generic to loop.
    """

    name = "convert-memref-stream-to-loops"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerGenericOpPattern()]),
            apply_recursively=False,
        ).rewrite_module(op)
