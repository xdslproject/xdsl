from xdsl.context import MLContext
from xdsl.dialects import bufferization, builtin, tensor
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class EmptyTensorLoweringPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: tensor.EmptyOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            bufferization.AllocTensorOp(
                op.tensor.type,
                op.dynamic_sizes,
            )
        )


class EmptyTensorToAllocTensorPass(ModulePass):
    """
    tensor.empty ops return a tensor of unspecified contents whose only purpose
    is to carry the tensor shape. This pass converts such ops to
    bufferization.alloc_tensor ops, which bufferize to buffer allocations.
    """

    name = "empty-tensor-to-alloc-tensor"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            EmptyTensorLoweringPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
