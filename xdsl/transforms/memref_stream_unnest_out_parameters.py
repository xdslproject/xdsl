from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ModuleOp,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class UnnestOutParametersPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if op.is_imperfectly_nested:
            # Already unnested
            return

        num_outputs = len(op.outputs)
        if not num_outputs:
            return

        num_inputs = len(op.inputs)

        num_parallel = sum(
            i == memref_stream.IteratorTypeAttr.parallel() for i in op.iterator_types
        )
        num_reduction = sum(
            i == memref_stream.IteratorTypeAttr.reduction() for i in op.iterator_types
        )
        if num_parallel == len(op.iterator_types):
            return

        parallel_dims = (True,) * num_parallel + (False,) * num_reduction

        maps = op.indexing_maps.data[num_inputs:]
        new_maps = ArrayAttr(
            (
                *op.indexing_maps.data[:num_inputs],
                *(AffineMapAttr(m.data.compress_dims(parallel_dims)) for m in maps),
            )
        )

        op.indexing_maps = new_maps


@dataclass(frozen=True)
class MemrefStreamUnnestOutParametersPass(ModulePass):
    """
    Converts the affine maps of memref_stream.generic out parameters from taking all the
    indices to only taking "parallel" ones.
    """

    name = "memref-stream-unnest-out-parameters"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            UnnestOutParametersPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
