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
from xdsl.ir import Attribute
from xdsl.utils.exceptions import LLVMTranslationException


def test_convert_int():
    with pytest.raises(NotImplementedError):
        convert_type(builtin.IntegerType(1))
    with pytest.raises(NotImplementedError):
        convert_type(builtin.IntegerType(32))
    with pytest.raises(NotImplementedError):
        convert_type(builtin.IntegerType(64))
    with pytest.raises(NotImplementedError):
        convert_type(builtin.IndexType())


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
    with pytest.raises(NotImplementedError):
        convert_type(vec_type)

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
    with pytest.raises(NotImplementedError):
        convert_type(arr_type)


def test_convert_struct():
    # Literal struct
    struct_type = LLVMStructType.from_type_list([builtin.i32, builtin.f32])
    with pytest.raises(NotImplementedError):
        convert_type(struct_type)

    # Tuple type -> struct
    tuple_type = builtin.TupleType(builtin.ArrayAttr([builtin.i32, builtin.f32]))
    with pytest.raises(NotImplementedError):
        convert_type(tuple_type)


def test_convert_function():
    # FunctionType
    func_type = builtin.FunctionType.from_lists(
        [builtin.i32, builtin.f32], [builtin.i64]
    )
    with pytest.raises(NotImplementedError):
        convert_type(func_type)

    # Void return
    func_void = builtin.FunctionType.from_lists([builtin.i32], [])
    with pytest.raises(NotImplementedError):
        convert_type(func_void)

    # LLVMFunctionType
    llvm_func = LLVMFunctionType([builtin.i32, builtin.f32], builtin.i64)
    with pytest.raises(NotImplementedError):
        convert_type(llvm_func)

    # Variadic LLVMFunction
    llvm_func_var = LLVMFunctionType([builtin.i32], builtin.i64, is_variadic=True)
    with pytest.raises(NotImplementedError):
        convert_type(llvm_func_var)


def test_unsupported_type():
    with pytest.raises(LLVMTranslationException, match="Type not supported"):
        convert_type(builtin.StringAttr("foo"))
