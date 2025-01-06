from dataclasses import dataclass, field
from itertools import repeat

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
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


def unroll_and_jam_bound_index(op: memref_stream.GenericOp) -> int | None:
    parallel_indices = tuple(
        index
        for index, iterator_type in enumerate(op.iterator_types)
        if iterator_type == memref_stream.IteratorTypeAttr.parallel()
    )
    if parallel_indices:
        return parallel_indices[-1]


def unroll_and_jam_interleave_factor(pipeline_depth: int, interleave_bound: int) -> int:
    interleave_factor = 1
    # Search factors until the next number divisible by pipeline_depth
    for potential_factor in range(pipeline_depth, pipeline_depth * 2):
        if not interleave_bound % potential_factor:
            # Found a larger factor
            interleave_factor = potential_factor
            break
    if interleave_factor == 1:
        # No larger perfect factors found, try smaller factors in descending order
        for potential_factor in range(pipeline_depth - 1, 1, -1):
            if not interleave_bound % potential_factor:
                # Found a smaller factor
                interleave_factor = potential_factor
                break

    return interleave_factor


def unroll_and_jam(
    op: memref_stream.GenericOp,
    rewriter: PatternRewriter,
    iterator_index: int,
    unroll_factor: int,
):
    if unroll_factor == 1:
        # If unroll factor is 1, rewrite is a no-op
        return

    old_block = op.body.block
    new_region = Region(
        Block(
            arg_types=(
                t for arg in old_block.args for t in repeat(arg.type, unroll_factor)
            )
        )
    )
    with ImplicitBuilder(new_region) as args:
        # For each interleaved block replica, a mapping from old values to new values
        value_map: tuple[dict[SSAValue, SSAValue], ...] = tuple(
            {} for _ in range(unroll_factor)
        )
        for arg_index, new_arg in enumerate(args):
            old_arg = old_block.args[arg_index // unroll_factor]
            value_map[arg_index % unroll_factor][old_arg] = new_arg
            new_arg.name_hint = old_arg.name_hint
        for block_op in old_block.ops:
            if isinstance(block_op, memref_stream.YieldOp):
                memref_stream.YieldOp(
                    *([vm[arg] for vm in value_map for arg in block_op.arguments])
                )
            else:
                for i in range(unroll_factor):
                    block_op.clone(value_mapper=value_map[i])

    # New maps are the same, except that they have one more dimension and the
    # dimension that is interleaved is updated to
    # `dim * interleave_factor + new_dim`.
    new_indexing_maps = ArrayAttr(
        AffineMapAttr(
            m.data.replace_dims_and_symbols(
                (
                    tuple(AffineExpr.dimension(i) for i in range(iterator_index))
                    + (
                        AffineExpr.dimension(iterator_index) * unroll_factor
                        + AffineExpr.dimension(m.data.num_dims),
                    )
                    + tuple(
                        AffineExpr.dimension(i)
                        for i in range(iterator_index + 1, m.data.num_dims + 2)
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
    new_bounds.append(IntegerAttr.from_index_int_value(unroll_factor))
    iterator_ub = op.bounds.data[iterator_index].value.data
    new_bounds[iterator_index] = IntegerAttr.from_index_int_value(
        iterator_ub // unroll_factor
    )

    rewriter.replace_matched_op(
        memref_stream.GenericOp(
            op.inputs,
            op.outputs,
            op.inits,
            new_region,
            new_indexing_maps,
            ArrayAttr(
                op.iterator_types.data + (memref_stream.IteratorTypeAttr.interleaved(),)
            ),
            ArrayAttr(new_bounds),
            op.init_indices,
            op.doc,
            op.library_call,
        )
    )


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

        interleave_bound_index = unroll_and_jam_bound_index(op)

        if interleave_bound_index is None:
            return

        interleave_bound = op.bounds.data[interleave_bound_index].value.data

        interleave_factor = unroll_and_jam_interleave_factor(
            self.pipeline_depth, interleave_bound
        )

        unroll_and_jam(op, rewriter, interleave_bound_index, interleave_factor)


@dataclass(frozen=True)
class MemrefStreamInterleavePass(ModulePass):
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

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            PipelineGenericPattern(self.pipeline_depth),
            apply_recursively=False,
        ).rewrite_module(op)
