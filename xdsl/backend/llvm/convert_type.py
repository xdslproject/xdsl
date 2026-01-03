from functools import cache

import llvmlite.ir as ir  # pyright: ignore[reportMissingTypeStubs]

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


@cache
def convert_type(type_attr: Attribute) -> ir.Type:
    """
    Convert an xDSL type attribute to an LLVM IR type.

    This function handles the conversion of various xDSL type attributes (integers, floats,
    pointers, vectors, arrays, structs, tuples, and functions) to their corresponding
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
    match type_attr:
        # Integers
        case IntegerType():
            return ir.IntType(type_attr.bitwidth)  # pyright: ignore[reportUnknownVariableType]
        case IndexType():
            return ir.IntType(64)  # pyright: ignore[reportUnknownVariableType]

        # Floats
        case Float16Type():
            return ir.HalfType()
        case Float32Type():
            return ir.FloatType()
        case Float64Type():
            return ir.DoubleType()

        # Void
        case LLVMVoidType() | NoneType():
            return ir.VoidType()

        # Pointers
        case LLVMPointerType():
            if isinstance(type_attr.addr_space, IntAttr):
                return ir.PointerType(type_attr.addr_space.data)
            return ir.PointerType()

        # Vectors and Arrays
        case VectorType():
            if type_attr.get_num_scalable_dims() > 0:
                raise LLVMTranslationException("Scalable vectors not supported")
            if type_attr.get_num_dims() != 1:
                raise LLVMTranslationException(
                    "Multi-dimensional vectors not supported"
                )
            return ir.VectorType(
                convert_type(type_attr.element_type),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                type_attr.get_shape()[0],
            )
        case LLVMArrayType():
            return ir.ArrayType(convert_type(type_attr.type), type_attr.size.data)

        # Aggregates
        case LLVMStructType():
            return ir.LiteralStructType([convert_type(t) for t in type_attr.types.data])
        case ComplexType():
            elem_type = convert_type(
                type_attr.element_type  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            return ir.LiteralStructType([elem_type, elem_type])
        case TupleType():
            return ir.LiteralStructType([convert_type(t) for t in type_attr.types.data])

        # Functions
        case FunctionType():
            return ir.FunctionType(
                convert_type(type_attr.outputs.data[0])
                if type_attr.outputs.data
                else ir.VoidType(),
                [convert_type(t) for t in type_attr.inputs.data],
            )
        case LLVMFunctionType():
            return ir.FunctionType(
                convert_type(type_attr.output),
                [convert_type(t) for t in type_attr.inputs.data],
                var_arg=type_attr.is_variadic,
            )

        case _:
            raise LLVMTranslationException(f"Type not supported: {type_attr}")
