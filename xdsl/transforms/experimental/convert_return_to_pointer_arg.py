from dataclasses import dataclass

from xdsl.dialects import func, llvm, builtin
from xdsl.ir import Block, Region, SSAValue
from xdsl.pattern_rewriter import RewritePattern, op_type_rewrite_pattern, PatternRewriter


@dataclass
class PassByPointerRewriter(RewritePattern):

    def __init__(self, func_names, originals: dict[builtin.StringAttr, func.FunctionType] = None):
        self.func_names = func_names
        self.originals = originals

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, func_op: func.FuncOp, rewriter: PatternRewriter
    ):
        if self.func_names is not None and not(func_op.sym_name in self.func_names or
                                               func_op.sym_name.data in self.func_names):
            return
        if self.originals is not None:
            if func_op.sym_name in self.originals:
                return
            func_op.update_function_type()
            self.originals[func_op.sym_name] = func_op.function_type
        if ((func_op.get_return_op() is None or len(func_op.get_return_op().arguments) == 0) and
                (all(not isinstance(arg.type, llvm.LLVMStructType) for arg in func_op.body.block.args))):
            return

        if func_op.get_return_op() is not None and len(func_op.get_return_op().arguments) > 0:
            body = func_op.body.block
            return_args = func_op.get_return_op().arguments

            for i, return_arg in enumerate(return_args):
                new_arg = body.insert_arg(llvm.LLVMPointerType.opaque(), i)
                rewriter.insert_op_before(llvm.StoreOp(return_arg, new_arg), func_op.get_return_op())

            rewriter.replace_op(func_op.get_return_op(), func.Return(), [])
            func_op.update_function_type()
            rewriter.handle_operation_modification(func_op)

        if any(isinstance(arg.type, llvm.LLVMStructType) for arg in func_op.body.block.args):
            body = func_op.body.block
            for arg in reversed(body.args):
                if isinstance(arg.type, llvm.LLVMStructType):
                    new_arg = body.insert_arg(llvm.LLVMPointerType.opaque(), arg.index)
                    body.insert_op_before(load_op := llvm.LoadOp(new_arg, arg.type), body.first_op)
                    arg.replace_by(load_op.dereferenced_value)
                    body.erase_arg(arg)
            func_op.update_function_type()
            rewriter.handle_operation_modification(func_op)



