import math
from typing import Any, Sequence, cast
from xdsl.backend.riscv.lowering.lower_utils import (
    cast_values_to_registers,
    get_type_size,
)
from xdsl.dialects.builtin import AnyFloat, ModuleOp, UnrealizedConversionCastOp
from xdsl.ir.core import Attribute, MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects import memref, riscv
from xdsl.transforms.dead_code_elimination import dce


def get_memref_size(memref: memref.MemRefType[Attribute]) -> int:
    return get_type_size(memref.element_type) * math.prod(memref.get_shape())


class LowerMemrefAlloc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Alloc is not supported")


class LowerMemrefAlloca(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloca, rewriter: PatternRewriter) -> None:
        if op.dynamic_sizes:
            raise NotImplementedError(
                f"Dynamically sized alloca is not supported {op.dynamic_sizes}"
            )

        reference_type = cast(memref.MemRefType[Attribute], op.memref.typ)
        total_size = get_memref_size(reference_type)

        rewriter.replace_matched_op(
            [
                sp := riscv.GetRegisterOp(riscv.Registers.SP),
                stack_alloc := riscv.AddiOp(sp, total_size),
                mv := riscv.MVOp(stack_alloc, rd=riscv.Registers.SP),
                UnrealizedConversionCastOp.get(mv.results, (reference_type,)),
            ]
        )


def insert_shape_ops(
    mem: SSAValue,
    element: memref.MemRefType[Attribute],
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
                size := riscv.LiOp(get_type_size(element)),
                idx := riscv.MulOp(indices[0], size),
                ptr := riscv.AddOp(mem, idx),
            ]
        )
    else:
        raise NotImplementedError(f"Shape {shape} is not supported")
    return ptr.rd


class LowerMemrefStore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter) -> None:
        value, mem, *indices = cast_values_to_registers(op.operands, rewriter)

        assert isinstance(op.memref.typ, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op.memref.typ)
        shape = memref_typ.get_shape()

        ptr = insert_shape_ops(mem, memref_typ.element_type, indices, shape, rewriter)
        rewriter.replace_matched_op(
            [
                mem_loc := riscv.AddOp(ptr, mem),
                riscv.FSwOp(
                    mem_loc, value, 0, comment=f"store value to memref of shape {shape}"
                )
                if isinstance(memref_typ.element_type, AnyFloat)
                else riscv.SwOp(
                    mem_loc, value, 0, comment=f"store value to memref of shape {shape}"
                ),
            ],
            [],
        )


class LowerMemrefLoad(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter) -> None:
        mem, *indices = cast_values_to_registers(op.operands, rewriter)

        assert isinstance(op.memref.typ, memref.MemRefType)
        memref_typ = cast(memref.MemRefType[Any], op.memref.typ)
        shape = memref_typ.get_shape()

        ptr = insert_shape_ops(mem, memref_typ.element_type, indices, shape, rewriter)
        rewriter.replace_matched_op(
            [
                mem_loc := riscv.AddOp(ptr, mem),
                lw := riscv.FLwOp(
                    mem_loc, 0, comment=f"load value from memref of shape {shape}"
                )
                if isinstance(memref_typ.element_type, AnyFloat)
                else riscv.LwOp(
                    mem_loc, 0, comment=f"load value from memref of shape {shape}"
                ),
                UnrealizedConversionCastOp.get(lw.results, (memref_typ.element_type,)),
            ],
            [],
        )


class RISCVStackMemrefLower(ModulePass):
    name = "lower-memref-to-stack-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerMemrefAlloc(),
                    LowerMemrefAlloca(),
                    LowerMemrefStore(),
                    LowerMemrefLoad(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)

        dce(op)
