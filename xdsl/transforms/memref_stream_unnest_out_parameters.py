from dataclasses import dataclass

from xdsl.dialects import linalg, memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ModuleOp,
)
from xdsl.ir import MLContext
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
        num_outputs = len(op.outputs)
        if not num_outputs:
            return
        if linalg.IteratorTypeAttr.window() in op.iterator_types:
            raise NotImplementedError()

        num_parallel = sum(
            i == linalg.IteratorTypeAttr.parallel() for i in op.iterator_types
        )
        num_reduction = sum(
            i == linalg.IteratorTypeAttr.reduction() for i in op.iterator_types
        )
        if num_parallel == len(op.iterator_types):
            return

        parallel_dims = (True,) * num_parallel + (False,) * num_reduction

        maps = op.indexing_maps.data[num_parallel:]
        new_maps = ArrayAttr(
            (
                *op.indexing_maps.data[:num_parallel],
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
