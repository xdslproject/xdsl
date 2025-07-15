from dataclasses import dataclass, field
from itertools import repeat

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    IntegerAttr,
    ModuleOp,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.ir.affine import AffineExpr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def factors(num: int) -> tuple[int, ...]:
    """
    For all positive integers, returns the n-tuple of all numbers that evenly divide the
    input, returns an empty tuple for 0 or negative inputs.
    """
    if num <= 0:
        return ()

    if num == 1:
        return (1,)

    return tuple(factor for factor in range(1, num + 1) if not num % factor)


@dataclass(frozen=True)
class PipelineGenericPattern(RewritePattern):
    pipeline_depth: int = field()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if memref_stream.IteratorTypeAttr.interleaved() in op.iterator_types:
            # Already interleaved
            return

        if memref_stream.IteratorTypeAttr.reduction() not in op.iterator_types:
            # No reduction
            return

        interleave_bound_index = -1
        interleave_bound = -1
        for index, (iterator_type, bound) in enumerate(
            zip(op.iterator_types, op.bounds, strict=True)
        ):
            if iterator_type == memref_stream.IteratorTypeAttr.parallel():
                interleave_bound = bound.value.data
                interleave_bound_index = index
        if interleave_bound == -1:
            # No parallel dimension
            return

        # Greatest number less than double of pipeline depth.
        # Want the biggest number for maximal instruction-level parallelism, less than
        # 2 * pipeline depth as a heuristic to limit register pressure.
        interleave_factor = max(
            f for f in factors(interleave_bound) if f < self.pipeline_depth * 2
        )

        old_block = op.body.block
        new_region = Region(
            Block(
                arg_types=(
                    t
                    for arg in old_block.args
                    for t in repeat(arg.type, interleave_factor)
                )
            )
        )
        with ImplicitBuilder(new_region) as args:
            # For each interleaved block replica, a mapping from old values to new values
            value_map: tuple[dict[SSAValue, SSAValue], ...] = tuple(
                {} for _ in range(interleave_factor)
            )
            for arg_index, new_arg in enumerate(args):
                old_arg = old_block.args[arg_index // interleave_factor]
                value_map[arg_index % interleave_factor][old_arg] = new_arg
                new_arg.name_hint = old_arg.name_hint
            for block_op in old_block.ops:
                if isinstance(block_op, memref_stream.YieldOp):
                    memref_stream.YieldOp(
                        *([vm[arg] for vm in value_map for arg in block_op.arguments])
                    )
                else:
                    for i in range(interleave_factor):
                        block_op.clone(value_mapper=value_map[i])

        # New maps are the same, except that they have one more dimension and the
        # dimension that is interleaved is updated to
        # `dim * interleave_factor + new_dim`.
        new_indexing_maps = ArrayAttr(
            AffineMapAttr(
                m.data.replace_dims_and_symbols(
                    (
                        tuple(
                            AffineExpr.dimension(i)
                            for i in range(interleave_bound_index)
                        )
                        + (
                            AffineExpr.dimension(interleave_bound_index)
                            * interleave_factor
                            + AffineExpr.dimension(m.data.num_dims),
                        )
                        + tuple(
                            AffineExpr.dimension(i)
                            for i in range(
                                interleave_bound_index + 1, m.data.num_dims + 2
                            )
                        )
                    ),
                    (),
                    m.data.num_dims + 1,
                    0,
                )
            )
            for m in op.indexing_maps
        )

        # The new bounds are the same, except there is one more bound
        new_bounds = list(op.bounds)
        new_bounds.append(IntegerAttr.from_index_int_value(interleave_factor))
        new_bounds[interleave_bound_index] = IntegerAttr.from_index_int_value(
            interleave_bound // interleave_factor
        )

        rewriter.replace_matched_op(
            memref_stream.GenericOp(
                op.inputs,
                op.outputs,
                op.inits,
                new_region,
                new_indexing_maps,
                ArrayAttr(
                    op.iterator_types.data
                    + (memref_stream.IteratorTypeAttr.interleaved(),)
                ),
                ArrayAttr(new_bounds),
                op.init_indices,
                op.doc,
                op.library_call,
            )
        )


@dataclass(frozen=True)
class MemRefStreamInterleavePass(ModulePass):
    """
    Tiles the innermost parallel dimension of a `memref_stream.generic`.
    If specified, the `pipeline-depth` parameter specifies the number of operations in the
    resulting body that should be executed concurrently.
    The pass will select the largest factor of the corresponding bound smaller than
    `pipeline-depth * 2`.
    The search range is bound by `pipeline-depth * 2` as very large interleaving factors
    can increase register pressure and potentially exhaust all available registers.
    In the future, it would be good to take the number of available registers into account
    when choosing a search range, as well as inspecting the generic body for
    read-after-write dependencies.
    """

    name = "memref-stream-interleave"

    pipeline_depth: int = field(default=4)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            PipelineGenericPattern(self.pipeline_depth),
            apply_recursively=False,
        ).rewrite_module(op)
