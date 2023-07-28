from math import prod
from typing import Any, Sequence, cast

from xdsl.dialects import memref, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir.core import MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from .lower_riscv_cf import cast_value_to_register


class LowerMemrefAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        assert isinstance(op.memref.type, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op.memref.type)
        size = prod(op.memref.type.get_shape())
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


def insert_shape_ops(
    mem: SSAValue,
    indices: Sequence[SSAValue],
    shape: Sequence[int],
    rewriter: PatternRewriter,
) -> SSAValue:
    """
    Returns ssa value representing pointer into the memref at given indices.
    """
    assert len(shape) == len(indices)

    if len(shape) == 1:
        rewriter.insert_op_before_matched_op(
            [
                ptr := riscv.AddOp(mem, indices[0]),
            ]
        )
    elif len(shape) == 2:
        rewriter.insert_op_before_matched_op(
            [
                cols := riscv.LiOp(shape[1]),
                row_offset := riscv.MulOp(cols, indices[0]),
                offset := riscv.AddOp(row_offset, indices[1]),
                word_bytes := riscv.LiOp(4),
                offset_bytes := riscv.MulOp(offset, word_bytes),
                ptr := riscv.AddOp(mem, offset_bytes),
            ]
        )
    else:
        assert False, f"Unsupported memref shape {shape}"
    return ptr.rd


class LowerMemrefStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter):
        value = cast_value_to_register(op.value, rewriter)
        mem = cast_value_to_register(op.memref, rewriter)
        indices = tuple(cast_value_to_register(index, rewriter) for index in op.indices)

        assert isinstance(op.memref.type, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op.memref.type)
        shape = memref_typ.get_shape()

        ptr = insert_shape_ops(mem, indices, shape, rewriter)
        rewriter.replace_matched_op(
            [
                riscv.SwOp(
                    ptr, value, 0, comment=f"store value to memref of shape {shape}"
                ),
            ]
        )


class LowerMemrefLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Load, rewriter: PatternRewriter):
        mem = cast_value_to_register(op.memref, rewriter)
        indices = tuple(cast_value_to_register(index, rewriter) for index in op.indices)

        assert isinstance(op.memref.type, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op.memref.type)
        shape = memref_typ.get_shape()
        ptr = insert_shape_ops(mem, indices, shape, rewriter)
        rewriter.replace_matched_op(
            [
                lw := riscv.LwOp(
                    ptr, 0, comment=f"load value from memref of shape {shape}"
                ),
                UnrealizedConversionCastOp.get(
                    lw.results, (riscv.IntRegisterType.unallocated(),)
                ),
            ],
        )


class LowerMemrefDeallocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Dealloc, rewriter: PatternRewriter):
        rewriter.erase_matched_op()


class LowerMemrefToRiscv(ModulePass):
    name = "lower-memref-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerMemrefAllocOp()).rewrite_module(op)
        PatternRewriteWalker(LowerMemrefStoreOp()).rewrite_module(op)
        PatternRewriteWalker(LowerMemrefLoadOp()).rewrite_module(op)
        PatternRewriteWalker(LowerMemrefDeallocOp()).rewrite_module(op)
