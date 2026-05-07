from typing import TypeAlias

from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class FuncRefType(ParametrizedAttribute, TypeAttribute):
    """
    Opaque type for function reference
    """

    name = "wasmssa.funcref"


@irdl_attr_definition
class ExternRefType(ParametrizedAttribute, TypeAttribute):
    """
    Opaque type for extern reference
    """

    name = "wasmssa.externref"


RefType: TypeAlias = FuncRefType | ExternRefType

WasmSSA = Dialect(
    "wasmssa",
    [],
    [
        ExternRefType,
        FuncRefType,
    ],
)
