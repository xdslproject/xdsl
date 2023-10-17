from xdsl.dialects import builtin
from xdsl.dialects.stencil import ApplyOp, BufferOp, StoreOp
from xdsl.ir import MLContext, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def should_materialize(temp: SSAValue):
    """
    Predicates if a specific stencil.apply output should be buffered.
    It should if it is used by another stencil.apply and not already buffered or stored.
    """
    return any(isinstance(u.operation, ApplyOp) for u in temp.uses) and not any(
        isinstance(u.operation, StoreOp | BufferOp) for u in temp.uses
    )


class ApplyOpMaterialization(RewritePattern):
    """
    Adds stencil.buffer to any used output of a stencil.apply that is not otherwised
    mapped to storage.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        clone = op.clone()
        new_res: list[OpResult] = []
        buffers: list[BufferOp] = []
        for i, out in enumerate(op.res):
            if should_materialize(out):
                buffer = BufferOp(clone.res[i])
                buffers.append(buffer)
                new_res.append(buffer.res)
            else:
                new_res.append(out)
        if buffers:
            rewriter.replace_matched_op([clone, *buffers], new_res)


class StencilStorageMaterializationPass(ModulePass):
    """
    Pass adding stencil.buffer whenever necessary to lower a stencil dialect IR,
    by adding stencil.buffer on any used stencil.apply output not otherwise mapped
    to storage.
    """

    name = "stencil-storage-materialization"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ApplyOpMaterialization(),
                ]
            )
        ).rewrite_module(op)
