import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]
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
from xdsl.utils.exceptions import LLVMTranslationException


def test_convert_int():
    assert convert_type(builtin.IntegerType(1)) == ir.IntType(1)
    assert convert_type(builtin.IntegerType(32)) == ir.IntType(32)
    assert convert_type(builtin.IntegerType(64)) == ir.IntType(64)
    assert convert_type(builtin.IndexType()) == ir.IntType(64)


def test_convert_float():
    assert convert_type(builtin.Float16Type()) == ir.HalfType()
    assert convert_type(builtin.Float32Type()) == ir.FloatType()
    assert convert_type(builtin.Float64Type()) == ir.DoubleType()


def test_convert_void():
    assert convert_type(builtin.NoneType()) == ir.VoidType()
    assert convert_type(LLVMVoidType()) == ir.VoidType()


def test_convert_pointer():
    # Test pointer without address space
    assert convert_type(LLVMPointerType()) == ir.PointerType()
    # Test with address space
    assert convert_type(LLVMPointerType(builtin.IntAttr(1))) == ir.PointerType(1)


def test_convert_vector():
    # 1D vector
    vec_type = builtin.VectorType(builtin.i32, [4])
    assert convert_type(vec_type) == ir.VectorType(ir.IntType(32), 4)  # pyright: ignore[reportUnknownArgumentType]

    # Scalable vector (raises)
    scalable_dims = builtin.ArrayAttr([builtin.IntegerAttr.from_bool(True)])
    scalable_vec = builtin.VectorType(builtin.i32, [4], scalable_dims)
    with pytest.raises(
        LLVMTranslationException, match="Scalable vectors not supported"
    ):
        convert_type(scalable_vec)

    # Multi-dimensional vector (raises)
    nd_vec = builtin.VectorType(builtin.i32, [2, 2])
    with pytest.raises(
        LLVMTranslationException, match="Multi-dimensional vectors not supported"
    ):
        convert_type(nd_vec)


def test_convert_array():
    arr_type = LLVMArrayType.from_size_and_type(10, builtin.i32)
    assert convert_type(arr_type) == ir.ArrayType(ir.IntType(32), 10)  # pyright: ignore[reportUnknownArgumentType]


def test_convert_struct():
    # Literal struct
    struct_type = LLVMStructType.from_type_list([builtin.i32, builtin.f32])
    expected = ir.LiteralStructType([ir.IntType(32), ir.FloatType()])  # pyright: ignore[reportUnknownArgumentType]
    assert convert_type(struct_type) == expected

    # Complex type -> struct { elem, elem }
    complex_type = builtin.ComplexType(builtin.f32)
    expected_complex = ir.LiteralStructType([ir.FloatType(), ir.FloatType()])
    assert convert_type(complex_type) == expected_complex

    # Tuple type -> struct
    tuple_type = builtin.TupleType(builtin.ArrayAttr([builtin.i32, builtin.f32]))
    assert convert_type(tuple_type) == expected


def test_convert_function():
    # FunctionType
    func_type = builtin.FunctionType.from_lists(
        [builtin.i32, builtin.f32], [builtin.i64]
    )
    expected_func = ir.FunctionType(ir.IntType(64), [ir.IntType(32), ir.FloatType()])  # pyright: ignore[reportUnknownArgumentType]
    assert convert_type(func_type) == expected_func

    # Void return
    func_void = builtin.FunctionType.from_lists([builtin.i32], [])
    expected_void = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])  # pyright: ignore[reportUnknownArgumentType]
    assert convert_type(func_void) == expected_void

    # LLVMFunctionType
    llvm_func = LLVMFunctionType([builtin.i32, builtin.f32], builtin.i64)
    assert convert_type(llvm_func) == expected_func

    # Variadic LLVMFunction
    llvm_func_var = LLVMFunctionType([builtin.i32], builtin.i64, is_variadic=True)
    expected_var = ir.FunctionType(ir.IntType(64), [ir.IntType(32)], var_arg=True)  # pyright: ignore[reportUnknownArgumentType]
    assert convert_type(llvm_func_var) == expected_var


def test_unsupported_type():
    with pytest.raises(LLVMTranslationException, match="Type not supported"):
        convert_type(builtin.StringAttr("foo"))
