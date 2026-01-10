from collections.abc import Callable
from functools import cache
from typing import Any

import llvmlite.ir as ir

from xdsl.dialects.builtin import (
    ComplexType,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerType,
    NoneType,
    TupleType,
    VectorType,
)
from xdsl.dialects.llvm import (
    LLVMArrayType,
    LLVMFunctionType,
    LLVMPointerType,
    LLVMStructType,
    LLVMVoidType,
)
from xdsl.ir import Attribute
from xdsl.utils.exceptions import LLVMTranslationException


def _convert_integer_type(type_attr: IntegerType) -> ir.Type:
    return ir.IntType(type_attr.width.data)


def _convert_index_type(_: IndexType) -> ir.Type:
    return ir.IntType(64)


def _convert_pointer_type(type_attr: LLVMPointerType) -> ir.Type:
    if isinstance(type_attr.addr_space, IntAttr):
        return ir.PointerType(type_attr.addr_space.data)
    return ir.PointerType()


def _convert_vector_type(type_attr: VectorType) -> ir.Type:
    if type_attr.get_num_scalable_dims() > 0:
        raise LLVMTranslationException("Scalable vectors not supported")
    if type_attr.get_num_dims() != 1:
        raise LLVMTranslationException("Multi-dimensional vectors not supported")
    return ir.VectorType(
        convert_type(type_attr.element_type),
        type_attr.get_shape()[0],
    )


def _convert_array_type(type_attr: LLVMArrayType) -> ir.Type:
    return ir.ArrayType(convert_type(type_attr.type), type_attr.size.data)


def _convert_struct_type(type_attr: LLVMStructType) -> ir.Type:
    return ir.LiteralStructType([convert_type(t) for t in type_attr.types.data])


def _convert_complex_type(type_attr: ComplexType) -> ir.Type:
    elem_type = convert_type(type_attr.element_type)
    return ir.LiteralStructType([elem_type, elem_type])


def _convert_tuple_type(type_attr: TupleType) -> ir.Type:
    return ir.LiteralStructType([convert_type(t) for t in type_attr.types.data])


def _convert_function_type(type_attr: FunctionType) -> ir.Type:
    return ir.FunctionType(
        convert_type(type_attr.outputs.data[0])
        if type_attr.outputs.data
        else ir.VoidType(),
        [convert_type(t) for t in type_attr.inputs.data],
    )


def _convert_llvm_function_type(type_attr: LLVMFunctionType) -> ir.Type:
    return ir.FunctionType(
        convert_type(type_attr.output),
        [convert_type(t) for t in type_attr.inputs.data],
        var_arg=type_attr.is_variadic,
    )


_TYPE_CONVERTERS: dict[type[Attribute], Callable[[Any], ir.Type]] = {
    IntegerType: _convert_integer_type,
    IndexType: _convert_index_type,
    Float16Type: lambda _: ir.HalfType(),
    Float32Type: lambda _: ir.FloatType(),
    Float64Type: lambda _: ir.DoubleType(),
    LLVMVoidType: lambda _: ir.VoidType(),
    NoneType: lambda _: ir.VoidType(),
    LLVMPointerType: _convert_pointer_type,
    VectorType: _convert_vector_type,
    LLVMArrayType: _convert_array_type,
    LLVMStructType: _convert_struct_type,
    ComplexType: _convert_complex_type,
    TupleType: _convert_tuple_type,
    FunctionType: _convert_function_type,
    LLVMFunctionType: _convert_llvm_function_type,
}


@cache
def convert_type(type_attr: Attribute) -> ir.Type:
    """
    Convert an xDSL type attribute to an LLVM IR type.

    This function handles the conversion of various xDSL type attributes (integers, floats,
    pointers, vectors, arrays, structs, tuples and functions) to their corresponding
    llvmlite IR type representations.

    Args:
        type_attr: The xDSL type attribute to convert.

    Returns:
        The corresponding llvmlite IR type.

    Raises:
        LLVMTranslationException: If the type is not supported, including:
            - Scalable vectors (vectors with scalable dimensions)
            - Multi-dimensional vectors (vectors with more than one dimension)
            - Any other unsupported type attribute
    """
    try:
        converter = _TYPE_CONVERTERS[type(type_attr)]
        return converter(type_attr)
    except KeyError:
        raise LLVMTranslationException(f"Type not supported: {type_attr}")
