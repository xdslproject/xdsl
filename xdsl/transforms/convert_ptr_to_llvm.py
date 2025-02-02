from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, func, llvm, ptr
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(llvm.StoreOp(op.value, op.addr))
        rewriter.insert_op(
            cast_op := builtin.UnrealizedConversionCastOp.get(
                [op.addr], [llvm.LLVMPointerType.opaque()]
            ),
            InsertPoint.before(op),
        )
        rewriter.replace_matched_op(llvm.StoreOp(op.value, cast_op.results[0]))


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter, /):
        rewriter.insert_op(
            cast_op := builtin.UnrealizedConversionCastOp.get(
                [op.addr], [llvm.LLVMPointerType.opaque()]
            ),
            InsertPoint.before(op),
        )
        rewriter.replace_matched_op(llvm.LoadOp(cast_op.results[0], op.res.type))


@dataclass
class ConvertFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        # rewrite function declaration
        new_input_types = [
            llvm.LLVMPointerType.opaque() if isinstance(arg, ptr.PtrType) else arg
            for arg in op.function_type.inputs
        ]
        new_output_types = [
            llvm.LLVMPointerType.opaque() if isinstance(arg, ptr.PtrType) else arg
            for arg in op.function_type.outputs
        ]
        op.function_type = func.FunctionType.from_lists(
            new_input_types,
            new_output_types,
        )

        if op.is_declaration:
            return

        insert_point = InsertPoint.at_start(op.body.blocks[0])

        # rewrite arguments
        for arg in op.args:
            if not isinstance(arg_type := arg.type, ptr.PtrType):
                continue

            arg.type = ptr.PtrType()

            if not arg.uses:
                continue

            rewriter.insert_op(
                cast_op := builtin.UnrealizedConversionCastOp.get([arg], [arg_type]),
                insert_point,
            )
            arg.replace_by_if(cast_op.results[0], lambda x: x.operation is not cast_op)


@dataclass
class ConvertReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter, /):
        if not any(isinstance(arg.type, ptr.PtrType) for arg in op.arguments):
            return

        insert_point = InsertPoint.before(op)
        new_arguments: list[SSAValue] = []

        # insert `ptr_xdsl.ptr -> llvm.ptr` casts for ptr return values
        for argument in op.arguments:
            if isinstance(argument.type, ptr.PtrType):
                rewriter.insert_op(
                    cast_op := builtin.UnrealizedConversionCastOp.get(
                        [argument], [llvm.LLVMPointerType.opaque()]
                    ),
                    insert_point,
                )
                new_arguments.append(cast_op.results[0])
            else:
                new_arguments.append(argument)

        rewriter.replace_matched_op(func.ReturnOp(*new_arguments))


class ConvertPtrToLLVMPass(ModulePass):
    name = "convert-ptr-to-llvm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertStoreOp(), ConvertLoadOp(), ConvertFuncOp(), ConvertReturnOp()]
            )
        ).rewrite_module(op)
