from typing import Annotated

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.irdl import IRDLOperation


class FieldType(ParametrizedAttribute):
    bounds: "StencilBoundsAttr | IntAttr"
    element_type: "Attribute"


class TempType(ParametrizedAttribute):
    bounds: "StencilBoundsAttr | IntAttr"
    element_type: "Attribute"


class ResultType(ParametrizedAttribute):
    elem: "Attribute"


class IndexAttr(ParametrizedAttribute):
    array: "Annotated[ArrayAttr[IntAttr], ArrayAttr]"


class StencilBoundsAttr(ParametrizedAttribute):
    lb: "IndexAttr"
    ub: "IndexAttr"


class CastOp(IRDLOperation):
    field: SSAValue


class CombineOp(IRDLOperation):
    lower: list[SSAValue]
    upper: list[SSAValue]
    lowerext: list[SSAValue]
    upperext: list[SSAValue]


class DynAccessOp(IRDLOperation):
    temp: SSAValue
    offset: list[SSAValue]


class ExternalLoadOp(IRDLOperation):
    field: SSAValue


class ExternalStoreOp(IRDLOperation):
    temp: SSAValue
    field: SSAValue


class IndexOp(IRDLOperation):
    pass


class AccessOp(IRDLOperation):
    temp: SSAValue


class LoadOp(IRDLOperation):
    field: SSAValue


class BufferOp(IRDLOperation):
    temp: SSAValue


class StoreOp(IRDLOperation):
    temp: SSAValue
    field: SSAValue


class ApplyOp(IRDLOperation):
    args: list[SSAValue]


class StoreResultOp(IRDLOperation):
    arg: list[SSAValue]


class ReturnOp(IRDLOperation):
    arg: list[SSAValue]


stencil: Dialect
