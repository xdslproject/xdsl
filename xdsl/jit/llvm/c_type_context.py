import ctypes

from xdsl.dialects.llvm import LLVMFunctionType, LLVMPointerType, LLVMVoidType
from xdsl.jit.c_type_context import CTypeContext
from xdsl.utils.exceptions import JITException


def register_llvm_ctypes(ctx: CTypeContext) -> None:
    ctx.register_ctype(LLVMPointerType, lambda _: ctypes.c_void_p)
    ctx.register_ctype(LLVMVoidType, lambda _: None)


def to_c_func_type(ctx: CTypeContext, func_type: LLVMFunctionType):
    """Build a ctypes function type from an LLVM function type."""
    if func_type.is_variadic:
        raise JITException(f"No ctypes mapping for variadic function type: {func_type}")
    return ctx.to_c_func_type(func_type.inputs, func_type.output)
