from collections.abc import Sequence
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
from xdsl.transforms.memref_stream_unroll_and_jam import (
    unroll_and_jam,
    unroll_and_jam_bound_indices_and_factors,
)


def interleave_index_and_factor(
    indices_and_factors: Sequence[tuple[int, int]], pipeline_depth: int
) -> tuple[int, int] | None:
    if not indices_and_factors:
        return None
    # Filter for innermost parallel index
    max_index = max(index for index, _ in indices_and_factors)
    indices_and_factors = tuple(
        (index, factor) for index, factor in indices_and_factors if index == max_index
    )

    # Reject factors greater than or equal to pipeline_depth * 2
    indices_and_factors = tuple(
        (index, factor)
        for index, factor in indices_and_factors
        if factor < pipeline_depth * 2
    )
    if not indices_and_factors:
        return None

    sorted_indices_and_factors = sorted(indices_and_factors, key=lambda x: x[1])

    return sorted_indices_and_factors[-1]


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

        indices_and_factors = unroll_and_jam_bound_indices_and_factors(op)
        if not indices_and_factors:
            return

        t = interleave_index_and_factor(indices_and_factors, self.pipeline_depth)
        if t is None:
            return

        index, factor = t

        unroll_and_jam(op, rewriter, index, factor)


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
