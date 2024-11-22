from __future__ import annotations

from typing import Generic, TypeVar

from xdsl.dialects.builtin import ContainerType
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    BaseAttr,
    GenericAttrConstraint,
    ParamAttrConstraint,
    ParameterDef,
    irdl_attr_definition,
)

_StreamTypeElement = TypeVar("_StreamTypeElement", bound=Attribute, covariant=True)
_StreamTypeElementConstrT = TypeVar("_StreamTypeElementConstrT", bound=Attribute)


class StreamType(
    Generic[_StreamTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
):
    element_type: ParameterDef[_StreamTypeElement]

    def __init__(self, element_type: _StreamTypeElement):
        super().__init__([element_type])

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type

    @staticmethod
    def constr(
        element_type: GenericAttrConstraint[_StreamTypeElementConstrT],
    ) -> ParamAttrConstraint[StreamType[_StreamTypeElementConstrT]]:
        return ParamAttrConstraint[StreamType[_StreamTypeElementConstrT]](
            StreamType, (element_type,)
        )


@irdl_attr_definition
class ReadableStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "stream.readable"

    @staticmethod
    def constr(
        element_type: GenericAttrConstraint[_StreamTypeElementConstrT],
    ) -> ParamAttrConstraint[ReadableStreamType[_StreamTypeElementConstrT]]:
        return ParamAttrConstraint[ReadableStreamType[_StreamTypeElementConstrT]](
            ReadableStreamType, (element_type,)
        )


AnyReadableStreamTypeConstr = BaseAttr[ReadableStreamType[Attribute]](
    ReadableStreamType
)


@irdl_attr_definition
class WritableStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "stream.writable"

    @staticmethod
    def constr(
        element_type: GenericAttrConstraint[_StreamTypeElementConstrT],
    ) -> ParamAttrConstraint[WritableStreamType[_StreamTypeElementConstrT]]:
        return ParamAttrConstraint[WritableStreamType[_StreamTypeElementConstrT]](
            WritableStreamType, (element_type,)
        )


AnyWritableStreamTypeConstr = BaseAttr[WritableStreamType[Attribute]](
    WritableStreamType
)


Stream = Dialect(
    "stream",
    [],
    [
        ReadableStreamType,
        WritableStreamType,
    ],
)
