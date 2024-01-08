from math import prod
from typing import Any, cast

from xdsl.dialects import memref, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerMemrefAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op_memref_type)
        size = prod(op_memref_type.get_shape())
        rewriter.replace_matched_op(
            [
                size_op := riscv.LiOp(size, comment="memref alloc size"),
                alloc_op := riscv.CustomAssemblyInstructionOp(
                    "buffer.alloc",
                    (size_op.rd,),
                    (riscv.IntRegisterType.unallocated(),),
                ),
                UnrealizedConversionCastOp.get(alloc_op.results, (memref_typ,)),
            ]
        )


class LowerMemrefDeallocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Dealloc, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class LowerMemrefToRiscv(ModulePass):
    name = "lower-memref-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerMemrefAllocOp()).rewrite_module(op)
        PatternRewriteWalker(LowerMemrefDeallocOp()).rewrite_module(op)
