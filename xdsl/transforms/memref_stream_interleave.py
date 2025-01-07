from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.memref_stream_unroll_and_jam import unroll_and_jam


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
