# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnnecessaryTypeIgnoreComment=false

import llvmlite.ir as ir
import pytest

from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import builtin
from xdsl.dialects.llvm import (
    LLVMArrayType,
    LLVMFunctionType,
    LLVMPointerType,
    LLVMStructType,
    LLVMVoidType,
)
from xdsl.ir import Attribute
from xdsl.utils.exceptions import LLVMTranslationException


def test_convert_int():
    assert convert_type(builtin.IntegerType(1)) == ir.IntType(1)
    assert convert_type(builtin.IntegerType(32)) == ir.IntType(32)
    assert convert_type(builtin.IntegerType(64)) == ir.IntType(64)
    assert convert_type(builtin.IndexType()) == ir.IntType(64)


@pytest.mark.parametrize(
    "type, expected",
    [
        (builtin.Float16Type(), ir.HalfType()),
        (builtin.Float32Type(), ir.FloatType()),
        (builtin.Float64Type(), ir.DoubleType()),
        (builtin.NoneType(), ir.VoidType()),
        (LLVMVoidType(), ir.VoidType()),
        (LLVMPointerType(), ir.PointerType()),
        (LLVMPointerType(builtin.IntAttr(1)), ir.PointerType(1)),
        (
            builtin.ComplexType(builtin.f32),
            ir.LiteralStructType([ir.FloatType(), ir.FloatType()]),
        ),
    ],
)
def test_convert_type(type: Attribute, expected: ir.Type):
    assert convert_type(type) == expected


def test_convert_vector():
    # 1D vector
    vec_type = builtin.VectorType(builtin.i32, [4])
    result = convert_type(vec_type)
    assert isinstance(result, ir.VectorType)
    assert result.count == 4
    assert result.element == ir.IntType(32)

    # Scalable vector (raises)
    scalable_dims = builtin.ArrayAttr([builtin.IntegerAttr.from_bool(True)])
    scalable_vec = builtin.VectorType(builtin.i32, [4], scalable_dims)
    with pytest.raises(
        LLVMTranslationException, match="Scalable vectors not supported"
    ):
        convert_type(scalable_vec)

    # Multi-dim vector (raises)
    nd_vec = builtin.VectorType(builtin.i32, [2, 2])
    with pytest.raises(
        LLVMTranslationException, match="Multi-dimensional vectors not supported"
    ):
        convert_type(nd_vec)


def test_convert_array():
    arr_type = LLVMArrayType.from_size_and_type(10, builtin.i32)
    result = convert_type(arr_type)
    assert isinstance(result, ir.ArrayType)
    assert result.count == 10
    assert result.element == ir.IntType(32)


def test_convert_struct():
    # Literal struct
    struct_type = LLVMStructType.from_type_list([builtin.i32, builtin.f32])
    result = convert_type(struct_type)
    assert isinstance(result, ir.LiteralStructType)
    assert result.elements == (ir.IntType(32), ir.FloatType())

    # Tuple type -> struct
    tuple_type = builtin.TupleType(builtin.ArrayAttr([builtin.i32, builtin.f32]))
    result = convert_type(tuple_type)
    assert isinstance(result, ir.LiteralStructType)
    assert result.elements == (ir.IntType(32), ir.FloatType())


def test_convert_function():
    # FunctionType
    func_type = builtin.FunctionType.from_lists(
        [builtin.i32, builtin.f32], [builtin.i64]
    )
    result = convert_type(func_type)
    assert isinstance(result, ir.FunctionType)
    assert result.return_type == ir.IntType(64)
    assert result.args == (ir.IntType(32), ir.FloatType())

    # Void return
    func_void = builtin.FunctionType.from_lists([builtin.i32], [])
    result = convert_type(func_void)
    assert isinstance(result, ir.FunctionType)
    assert result.return_type == ir.VoidType()
    assert result.args == (ir.IntType(32),)

    # LLVMFunctionType
    llvm_func = LLVMFunctionType([builtin.i32, builtin.f32], builtin.i64)
    result = convert_type(llvm_func)
    assert isinstance(result, ir.FunctionType)
    assert result.return_type == ir.IntType(64)
    assert result.args == (ir.IntType(32), ir.FloatType())
    assert result.var_arg is False

    # Variadic LLVMFunction
    llvm_func_var = LLVMFunctionType([builtin.i32], builtin.i64, is_variadic=True)
    result = convert_type(llvm_func_var)
    assert isinstance(result, ir.FunctionType)
    assert result.return_type == ir.IntType(64)
    assert result.args == (ir.IntType(32),)
    assert result.var_arg is True


def test_unsupported_type():
    with pytest.raises(LLVMTranslationException, match="Type not supported"):
        convert_type(builtin.StringAttr("foo"))
