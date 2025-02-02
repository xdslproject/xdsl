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

            arg.type = llvm.LLVMPointerType.opaque()

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


@dataclass
class ConvertCallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter, /):
        if not any(
            isinstance(arg.type, ptr.PtrType) for arg in op.arguments
        ) and not any(isinstance(type, ptr.PtrType) for type in op.result_types):
            return

        # rewrite arguments
        insert_point = InsertPoint.before(op)
        new_arguments: list[SSAValue] = []

        # insert `ptr_xdsl.ptr -> llvm.ptr` casts for argument values
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

        insert_point = InsertPoint.after(op)
        new_results: list[SSAValue] = []

        #  insert `llvm.ptr -> ptr_xdsl.ptr` casts for return values
        for result in op.results:
            if isinstance(result.type, ptr.PtrType):
                rewriter.insert_op(
                    cast_op := builtin.UnrealizedConversionCastOp.get(
                        [result],
                        [result.type],
                    ),
                    insert_point,
                )
                new_results.append(cast_op.results[0])
            else:
                new_results.append(result)

        new_return_types = [
            llvm.LLVMPointerType.opaque() if isinstance(type, ptr.PtrType) else type
            for type in op.result_types
        ]

        rewriter.replace_matched_op(
            func.CallOp(op.callee, new_arguments, new_return_types)
        )


class ConvertPtrAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(llvm.AddOp(op.addr, op.offset))


class ReconcileUnrealizedPtrCasts(RewritePattern):
    """
    Eliminates`llvm.ptr -> ptr_xdsl.ptr -> llvm.ptr` casts.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: builtin.UnrealizedConversionCastOp, rewriter: PatternRewriter, /
    ):
        # preconditions
        if (
            len(op.inputs) != 1
            or len(op.outputs) != 1
            or not isinstance(op.inputs[0].type, llvm.LLVMPointerType)
            or not isinstance(op.outputs[0].type, ptr.PtrType)
        ):
            return

        # erase ptr -> memref -> ptr cast pairs
        uses = tuple(use for use in op.outputs[0].uses)
        for use in uses:
            if (
                isinstance(use.operation, builtin.UnrealizedConversionCastOp)
                and isinstance(use.operation.inputs[0].type, ptr.PtrType)
                and isinstance(use.operation.outputs[0].type, llvm.LLVMPointerType)
            ):
                use.operation.outputs[0].replace_by(op.inputs[0])
                rewriter.erase_op(use.operation)

        # erase this cast entirely if no uses are remaining
        if op.outputs[0].uses:
            return

        rewriter.erase_op(op)


class ConvertPtrToLLVMPass(ModulePass):
    name = "convert-ptr-to-llvm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStoreOp(),
                    ConvertLoadOp(),
                    ConvertFuncOp(),
                    ConvertReturnOp(),
                    ConvertCallOp(),
                    ReconcileUnrealizedPtrCasts(),
                ]
            )
        ).rewrite_module(op)
