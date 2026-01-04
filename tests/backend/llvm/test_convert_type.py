import pytest
from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]

from xdsl.backend.llvm.convert import LLVMTranslationException, convert_type
from xdsl.dialects.builtin import Float64Type, IntegerType, UnrankedTensorType
from xdsl.dialects.llvm import LLVMVoidType


def test_convert_float64():
    assert convert_type(Float64Type()) == ir.DoubleType()


def test_convert_void():
    assert convert_type(LLVMVoidType()) == ir.VoidType()


def test_convert_integer():
    assert convert_type(IntegerType(32)) == ir.IntType(32)
    assert convert_type(IntegerType(64)) == ir.IntType(64)
    assert convert_type(IntegerType(1)) == ir.IntType(1)


def test_convert_unsupported():
    with pytest.raises(LLVMTranslationException):
        convert_type(UnrankedTensorType(Float64Type()))
