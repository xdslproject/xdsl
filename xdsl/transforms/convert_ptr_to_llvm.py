from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, ptr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            (
                cast_op := builtin.UnrealizedConversionCastOp.get(
                    (op.addr,), (llvm.LLVMPointerType(),)
                ),
                llvm.StoreOp(op.value, cast_op.results[0]),
            ),
        )


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            (
                cast_op := builtin.UnrealizedConversionCastOp.get(
                    [op.addr], [llvm.LLVMPointerType()]
                ),
                llvm.LoadOp(cast_op.results[0], op.res.type),
            ),
        )


@dataclass
class ConvertPtrAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            (
                cast_addr_op := builtin.UnrealizedConversionCastOp.get(
                    [op.addr],
                    [llvm.LLVMPointerType()],
                ),
                # offset (index) -> offset (int)
                offest_to_int_op := arith.IndexCastOp(op.offset, builtin.i64),
                # ptr -> int
                ptr_to_int_op := llvm.PtrToIntOp(
                    cast_addr_op.results[0],
                    builtin.i64,
                ),
                # int + arg
                add_op := arith.AddiOp(
                    ptr_to_int_op.results[0], offest_to_int_op.result
                ),
                # int -> ptr
                llvm.IntToPtrOp(add_op.result),
            ),
        )


class ConvertToPtrOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.ToPtrOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, (), op.operands)


class ConvertFromPtrOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.FromPtrOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, (), op.operands)


class RewritePtrTypes(TypeConversionPattern):
    """
    Replaces `ptr_dxdsl.ptr` with `llvm.ptr`.
    """

    @attr_type_rewrite_pattern
    def convert_type(self, typ: ptr.PtrType):
        return llvm.LLVMPointerType()


class ConvertPtrToLLVMPass(ModulePass):
    name = "convert-ptr-to-llvm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStoreOp(),
                    ConvertLoadOp(),
                    ConvertPtrAddOp(),
                    ConvertToPtrOp(),
                    ConvertFromPtrOp(),
                    RewritePtrTypes(recursive=True),
                ]
            )
        ).rewrite_module(op)
