from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import repeat
from typing import NamedTuple

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
from xdsl.utils.op_selector import OpSelector


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


class IndexAndFactor(NamedTuple):
    """
    Helper data structure holding an option for which index of a `memref_stream.generic`
    operation to interleave and with which factor.
    """

    iterator_index: int
    factor: int

    @staticmethod
    def choose(
        indices_and_factors: Sequence[IndexAndFactor], pipeline_depth: int
    ) -> IndexAndFactor | None:
        """
        A heuristic to choose the interleave index and factor automatically given the
        pipeline depth of floating-point operations on the processor.
        The higher the factor chosen, the higher the instruction-level parallelism, but
        also the more registers need to be used at the same time, potentially leading to
        spilling.
        """
        if not indices_and_factors:
            return None
        # Filter for innermost parallel index
        max_index = max(index for index, _ in indices_and_factors)
        indices_and_factors = tuple(
            t for t in indices_and_factors if t.iterator_index == max_index
        )

        # Want the biggest number for maximal instruction-level parallelism, less than
        # 2 * pipeline depth as a heuristic to limit register pressure.
        indices_and_factors = tuple(
            t for t in indices_and_factors if t.factor < pipeline_depth * 2
        )
        if not indices_and_factors:
            return None

        sorted_indices_and_factors = sorted(indices_and_factors, key=lambda x: x[1])

        # Greatest number less than double of pipeline depth.
        return sorted_indices_and_factors[-1]


@dataclass(frozen=True)
class PipelineGenericPattern(RewritePattern):
    pipeline_depth: int
    iterator_index: int | None = field(default=None)
    unroll_factor: int | None = field(default=None)

    @staticmethod
    def indices_and_factors(
        op: memref_stream.GenericOp,
    ) -> tuple[IndexAndFactor, ...]:
        """
        Given a `memref_stream.generic` operation, returns all the possible options for
        unrolling.
        """
        if memref_stream.IteratorTypeAttr.interleaved() in op.iterator_types:
            # Already interleaved
            return ()
        parallel_indices = tuple(
            index
            for index, iterator_type in enumerate(op.iterator_types)
            if iterator_type == memref_stream.IteratorTypeAttr.parallel()
        )
        parallel_bounds = tuple(
            op.bounds.data[index].value.data for index in parallel_indices
        )
        return tuple(
            IndexAndFactor(index, factor)
            for index, bound in zip(parallel_indices, parallel_bounds)
            for factor in factors(bound)
        )

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

        assert (self.iterator_index is None) == (self.unroll_factor is None)

        if self.iterator_index is not None and self.unroll_factor is not None:
            interleave_bound_index = self.iterator_index
            interleave_factor = self.unroll_factor
        else:
            indices_and_factors = self.indices_and_factors(op)
            if not indices_and_factors:
                return

            t = IndexAndFactor.choose(indices_and_factors, self.pipeline_depth)
            if t is None:
                return

            interleave_bound_index, interleave_factor = t

        if interleave_factor == 1:
            # If unroll factor is 1, rewrite is a no-op
            return

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
        interleave_bound = op.bounds.data[interleave_bound_index].value.data
        new_bounds[interleave_bound_index] = IntegerAttr.from_index_int_value(
            interleave_bound // interleave_factor
        )

        rewriter.replace_op(
            op,
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
            ),
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
    op_index: int | None = field(default=None)
    iterator_index: int | None = field(default=None)
    unroll_factor: int | None = field(default=None)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pattern = PipelineGenericPattern(
            self.pipeline_depth,
            self.iterator_index,
            self.unroll_factor,
        )
        if self.op_index is not None:
            matched_op = OpSelector(self.op_index, "memref_stream.generic").get_op(op)
            pattern.match_and_rewrite(matched_op, PatternRewriter(matched_op))
            return

        PatternRewriteWalker(pattern, apply_recursively=False).rewrite_module(op)

    @classmethod
    def schedule_space(cls, ctx: Context, module_op: ModuleOp):
        return tuple(
            MemRefStreamInterleavePass(
                op_index=op_idx,
                iterator_index=iterator_index,
                unroll_factor=unroll_factor,
            )
            for op_idx, matched_op in enumerate(module_op.walk())
            if isinstance(matched_op, memref_stream.GenericOp)
            for iterator_index, unroll_factor in PipelineGenericPattern.indices_and_factors(
                matched_op
            )
        )
