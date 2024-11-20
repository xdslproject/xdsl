from __future__ import annotations

import abc
from typing import ClassVar, Generic, TypeVar, cast

from xdsl.dialects.builtin import ContainerType
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    BaseAttr,
    GenericAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    ParameterDef,
    VarConstraint,
    irdl_attr_definition,
    operand_def,
    result_def,
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


class ReadOperation(IRDLOperation, abc.ABC):
    """
    Abstract base class for operations that read from a stream.
    """

    T: ClassVar = VarConstraint("T", AnyAttr())

    stream = operand_def(ReadableStreamType.constr(T))
    res = result_def(T)

    assembly_format = "`from` $stream attr-dict `:` type($res)"

    def __init__(self, stream: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream_type := stream.type, ReadableStreamType)
            stream_type = cast(ReadableStreamType[Attribute], stream_type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream], result_types=[result_type])


class WriteOperation(IRDLOperation, abc.ABC):
    """
    Abstract base class for operations that write to a stream.
    """

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    stream = operand_def(WritableStreamType.constr(T))

    assembly_format = "$value `to` $stream attr-dict `:` type($value)"

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])


Stream = Dialect(
    "stream",
    [],
    [
        ReadableStreamType,
        WritableStreamType,
    ],
)
