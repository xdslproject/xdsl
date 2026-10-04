from xdsl.dialects.llvm import LLVMFunctionType, LLVMPointerType, LLVMVoidType
from xdsl.jit.c_type_context import CTypeContext
from xdsl.utils.exceptions import JITException


def register_llvm_types(ctx: CTypeContext) -> None:
    ctx.register_type(LLVMPointerType, lambda _: "void *")
    ctx.register_type(LLVMVoidType, lambda _: "void")


def to_c_func_type(ctx: CTypeContext, func_type: LLVMFunctionType):
    """Build a C function signature from an LLVM function type."""
    if func_type.is_variadic:
        raise JITException(f"Variadic function types are not supported: {func_type}")
    return ctx.to_c_func_type(func_type.inputs, func_type.output)
