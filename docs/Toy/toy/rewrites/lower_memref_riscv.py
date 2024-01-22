from math import prod
from typing import Any, cast

from xdsl.dialects import memref, riscv, riscv_func
from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr, UnrealizedConversionCastOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable


class InsertLibcOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
        if any(isinstance(o, memref.Alloc) for o in op.walk()):
            # Add external putchar reference if not already there
            if SymbolTable.lookup_symbol(op, "malloc") is None:
                op.body.block.add_ops(
                    [
                        riscv_func.FuncOp.external(
                            "malloc",
                            [riscv.IntRegisterType.a_register(0)],
                            [riscv.IntRegisterType.a_register(0)],
                        )
                    ]
                )
        if any(isinstance(o, memref.Dealloc) for o in op.walk()):
            # Add external putchar reference if not already there
            if SymbolTable.lookup_symbol(op, "free") is None:
                op.body.block.add_ops(
                    [
                        riscv_func.FuncOp.external(
                            "free",
                            [riscv.IntRegisterType.a_register(0)],
                            [riscv.IntRegisterType.a_register(0)],
                        )
                    ]
                )


class LowerMemrefAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op_memref_type)
        size = prod(op_memref_type.get_shape())
        rewriter.replace_matched_op(
            [
                size_op := riscv.LiOp(size * 8, comment="memref alloc size"),
                alloc_op := riscv_func.CallOp(
                    SymbolRefAttr("malloc"),
                    (size_op.rd,),
                    (riscv.IntRegisterType.a_register(0),),
                ),
                mv_op := riscv.MVOp(
                    alloc_op.results[0], rd=riscv.IntRegisterType.unallocated()
                ),
                UnrealizedConversionCastOp.get((mv_op.rd,), (memref_typ,)),
            ]
        )


class LowerMemrefDeallocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Dealloc, rewriter: PatternRewriter):
        assert isinstance(op.memref.type, memref.MemRefType)
        rewriter.replace_matched_op(
            [
                cast_op := UnrealizedConversionCastOp.get(
                    (op.memref,), (riscv.IntRegisterType.unallocated(),)
                ),
                mv_op := riscv.MVOp(
                    cast_op.results[0], rd=riscv.IntRegisterType.a_register(0)
                ),
                riscv_func.CallOp(
                    SymbolRefAttr("free"),
                    (mv_op.rd,),
                    (),
                ),
            ]
        )


class LowerMemrefToRiscv(ModulePass):
    name = "lower-memref-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    InsertLibcOps(),
                    LowerMemrefAllocOp(),
                    LowerMemrefDeallocOp(),
                ]
            ),
        ).rewrite_module(op)
