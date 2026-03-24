from typing import TypeAlias

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    I32,
    I64,
    I128,
    ContainerType,
    Float32Type,
    Float64Type,
    FunctionType,
    IntAttr,
)
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition

IntegerType: TypeAlias = I32 | I64
FPType: TypeAlias = Float32Type | Float64Type
NumericType: TypeAlias = IntegerType | FPType
VecType: TypeAlias = I128


@irdl_attr_definition
class FuncRefType(TypeAttribute):
    """
    Opaque type for function reference
    """

    name = "funcref"


@irdl_attr_definition
class ExternRefType(TypeAttribute):
    """
    Opaque type for extern reference
    """

    name = "externref"


RefType: TypeAlias = FuncRefType | ExternRefType
ValType: TypeAlias = NumericType | VecType | RefType
ResultType: TypeAlias = ContainerType[ValType]
FuncType: TypeAlias = FunctionType


@irdl_attr_definition
class LimitType(TypeAttribute):
    """
    Wasm limit type
    """

    name = "limit"

    min: IntAttr
    max: IntAttr | None


@irdl_attr_definition
class LocalRefType(ParametrizedAttribute, TypeAttribute):
    """
    Type of a local variable
    """

    name = "local"

    elementType: ValType


@irdl_attr_definition
class TableType(ParametrizedAttribute):
    """
    Wasm table type
    """

    name = "tabletype"

    reference: RefType
    limit: LimitType


IntegerAttr = builtin.IntegerAttr[I32] | builtin.IntegerAttr[I64]
FPAttr = builtin.FloatAttr[Float32Type] | builtin.FloatAttr[Float64Type]
NumericAttr = IntegerAttr | FPAttr

WasmSSA = Dialect(
    "wasmssa",
    [],
    [
        FuncType,
        LimitType,
        TableType,
    ],
)
