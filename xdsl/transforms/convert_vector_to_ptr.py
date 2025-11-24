from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, memref, ptr, vector
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.convert_memref_to_ptr import get_target_ptr
from xdsl.utils.hints import isa


@dataclass
class VectorStoreToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.StoreOp, rewriter: PatternRewriter):
        assert isa(memref_type := op.base.type, memref.MemRefType)
        target_ptr = get_target_ptr(op.base, memref_type, op.indices, rewriter)
        rewriter.replace_op(op, ptr.StoreOp(addr=target_ptr, value=op.vector))


@dataclass
class VectorLoadToPtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.LoadOp, rewriter: PatternRewriter):
        assert isa(memref_type := op.base.type, memref.MemRefType)
        target_ptr = get_target_ptr(op.base, memref_type, op.indices, rewriter)
        rewriter.replace_op(op, ptr.LoadOp(target_ptr, op.result.type))


@dataclass(frozen=True)
class ConvertVectorToPtrPass(ModulePass):
    name = "convert-vector-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    VectorLoadToPtr(),
                    VectorStoreToPtr(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
