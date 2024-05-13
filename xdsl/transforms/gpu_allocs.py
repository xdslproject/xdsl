from xdsl.dialects import gpu, memref
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class GpuAllocPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            gpu.AllocOp.build(
                operands=[None, op.dynamic_sizes, op.symbol_operands],
                result_types=[op.memref.type, None],
            )
        )


class GpuDellocPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Dealloc, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(gpu.DeallocOp(op.memref))


class MemrefToGPUPass(ModulePass):
    name = "memref-to-gpu"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GpuAllocPattern(),
                    GpuDellocPattern(),
                ]
            )
        )
        walker.rewrite_module(op)
