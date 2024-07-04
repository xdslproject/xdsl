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
from xdsl.utils.exceptions import DiagnosticException


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

        if interleave_bound % self.pipeline_depth:
            raise DiagnosticException(
                "Ranges of interleave factors not implemented yet"
            )

        old_block = op.body.block
        new_region = Region(
            Block(
                arg_types=(
                    t
                    for arg in old_block.args
                    for t in repeat(arg.type, self.pipeline_depth)
                )
            )
        )
        with ImplicitBuilder(new_region) as args:
            # For each interleaved block replica, a mapping from old values to new values
            value_map: tuple[dict[SSAValue, SSAValue], ...] = tuple(
                {} for _ in range(self.pipeline_depth)
            )
            for arg_index, new_arg in enumerate(args):
                old_arg = old_block.args[arg_index // self.pipeline_depth]
                value_map[arg_index % self.pipeline_depth][old_arg] = new_arg
                new_arg.name_hint = old_arg.name_hint
            for block_op in old_block.ops:
                if isinstance(block_op, memref_stream.YieldOp):
                    memref_stream.YieldOp(
                        *([vm[arg] for vm in value_map for arg in block_op.arguments])
                    )
                else:
                    for i in range(self.pipeline_depth):
                        block_op.clone(value_mapper=value_map[i])

        # New maps are the same, except that they have one more dimension and the
        # dimension that is interleaved is updated to
        # `dim * self.pipeline_depth + new_dim`.
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
                            * self.pipeline_depth
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
        new_bounds.append(IntegerAttr.from_index_int_value(self.pipeline_depth))
        new_bounds[interleave_bound_index] = IntegerAttr.from_index_int_value(
            interleave_bound // self.pipeline_depth
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
            )
        )


@dataclass(frozen=True)
class MemrefStreamInterleavePass(ModulePass):
    """
    Tiles the innermost parallel dimension of a `memref_stream.generic`.
    If specified, the `pipeline-depth` parameter specifies the number of operations in the
    resulting body that should be executed concurrently.
    """

    name = "memref-stream-interleave"

    pipeline_depth: int = field(default=4)

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            PipelineGenericPattern(self.pipeline_depth),
            apply_recursively=False,
        ).rewrite_module(op)
