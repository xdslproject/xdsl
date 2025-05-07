from dataclasses import dataclass

from xdsl.context import Context
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

        reduction_dims = (False,) * num_parallel + (True,) * num_reduction

        maps = op.indexing_maps.data[num_inputs:]
        new_maps = ArrayAttr(
            (
                *op.indexing_maps.data[:num_inputs],
                *(AffineMapAttr(m.data.drop_dims(reduction_dims)) for m in maps),
            )
        )

        op.indexing_maps = new_maps


@dataclass(frozen=True)
class MemRefStreamUnnestOutParametersPass(ModulePass):
    """
    Converts the affine maps of memref_stream.generic out parameters from taking all the
    indices to only taking "parallel" ones.
    """

    name = "memref-stream-unnest-out-parameters"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            UnnestOutParametersPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
