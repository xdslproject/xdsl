import ctypes

import pytest

from xdsl.dialects.builtin import Float64Type, IntAttr, IntegerType
from xdsl.dialects.llvm import LLVMFunctionType, LLVMPointerType, LLVMVoidType
from xdsl.ir import Attribute
from xdsl.jit.c_type_context import CTypeContext, register_builtin_ctypes
from xdsl.jit.llvm.c_type_context import register_llvm_ctypes, to_c_func_type
from xdsl.utils.exceptions import JITException


@pytest.fixture
def ctx() -> CTypeContext:
    c = CTypeContext()
    register_builtin_ctypes(c)
    register_llvm_ctypes(c)
    return c


@pytest.mark.parametrize(
    "type_attr, expected",
    [
        (LLVMPointerType(), ctypes.c_void_p),
        (LLVMPointerType(IntAttr(1)), ctypes.c_void_p),
        (LLVMVoidType(), None),
    ],
)
def test_llvm_resolve(ctx: CTypeContext, type_attr: Attribute, expected: object):
    assert ctx.to_ctype(type_attr) is expected


@pytest.mark.parametrize("type_attr", [LLVMPointerType(), LLVMVoidType()])
def test_llvm_types_are_not_registered_by_builtin(type_attr: Attribute):
    ctx = CTypeContext()
    register_builtin_ctypes(ctx)
    with pytest.raises(JITException, match="No ctypes mapping"):
        ctx.to_ctype(type_attr)


def test_to_c_func_type(ctx: CTypeContext):
    func_type = LLVMFunctionType((Float64Type(), IntegerType(32)), Float64Type())
    assert to_c_func_type(ctx, func_type) is ctypes.CFUNCTYPE(
        ctypes.c_double, ctypes.c_double, ctypes.c_int32
    )


def test_to_c_func_type_void_no_args(ctx: CTypeContext):
    assert to_c_func_type(ctx, LLVMFunctionType(())) is ctypes.CFUNCTYPE(None)


def test_to_c_func_type_variadic_raises(ctx: CTypeContext):
    func_type = LLVMFunctionType((Float64Type(),), Float64Type(), is_variadic=True)
    with pytest.raises(JITException, match="variadic function type"):
        to_c_func_type(ctx, func_type)
