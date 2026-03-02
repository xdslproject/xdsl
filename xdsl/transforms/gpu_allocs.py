from xdsl.context import Context
from xdsl.dialects import gpu, memref
from xdsl.dialects.builtin import ModuleOp
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
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            gpu.AllocOp.build(
                operands=[None, op.dynamic_sizes, op.symbol_operands],
                result_types=[op.memref.type, None],
            ),
        )


class GpuDellocPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.DeallocOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, gpu.DeallocOp(op.memref))


class MemRefToGPUPass(ModulePass):
    name = "memref-to-gpu"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GpuAllocPattern(),
                    GpuDellocPattern(),
                ]
            )
        )
        walker.rewrite_module(op)
