import pytest

from xdsl.dialects.builtin import Float64Type, IntAttr, IntegerType
from xdsl.dialects.llvm import LLVMFunctionType, LLVMPointerType, LLVMVoidType
from xdsl.ir import Attribute
from xdsl.jit.c_type_context import CFuncType, CTypeContext, register_builtin_types
from xdsl.jit.llvm.c_type_context import register_llvm_types, to_c_func_type
from xdsl.utils.exceptions import JITException


@pytest.fixture
def ctx() -> CTypeContext:
    c = CTypeContext()
    register_builtin_types(c)
    register_llvm_types(c)
    return c


@pytest.mark.parametrize(
    "type_attr, expected",
    [
        (LLVMPointerType(), "void *"),
        (LLVMPointerType(IntAttr(1)), "void *"),
        (LLVMVoidType(), "void"),
    ],
)
def test_llvm_resolve(ctx: CTypeContext, type_attr: Attribute, expected: str):
    assert ctx.to_type(type_attr) == expected


@pytest.mark.parametrize("type_attr", [LLVMPointerType(), LLVMVoidType()])
def test_llvm_types_are_not_registered_by_builtin(type_attr: Attribute):
    ctx = CTypeContext()
    register_builtin_types(ctx)
    with pytest.raises(JITException, match="No C type mapping"):
        ctx.to_type(type_attr)


def test_to_c_func_type(ctx: CTypeContext):
    func_type = LLVMFunctionType((Float64Type(), IntegerType(32)), Float64Type())
    assert to_c_func_type(ctx, func_type) == CFuncType(("double", "int32_t"), "double")


def test_to_c_func_type_void_no_args(ctx: CTypeContext):
    assert to_c_func_type(ctx, LLVMFunctionType(())) == CFuncType((), "void")


def test_to_c_func_type_variadic_raises(ctx: CTypeContext):
    func_type = LLVMFunctionType((Float64Type(),), Float64Type(), is_variadic=True)
    with pytest.raises(JITException, match="Variadic function types"):
        to_c_func_type(ctx, func_type)
