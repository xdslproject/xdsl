from __future__ import annotations

import abc
from typing import ClassVar, Generic, TypeAlias, TypeVar, cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    ContainerType,
)
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
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer

_StreamTypeElement = TypeVar("_StreamTypeElement", bound=Attribute)


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

    @classmethod
    def constr(
        cls,
        *,
        element_type: GenericAttrConstraint[_StreamTypeElement] = AnyAttr(),
    ) -> GenericAttrConstraint[StreamType[_StreamTypeElement]]:
        if element_type == AnyAttr():
            return BaseAttr[StreamType[_StreamTypeElement]](StreamType)
        return ParamAttrConstraint[StreamType[_StreamTypeElement]](
            StreamType, (element_type,)
        )


@irdl_attr_definition
class ReadableStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "stream.readable"


@irdl_attr_definition
class WritableStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "stream.writable"


AnyWritableStreamType: TypeAlias = WritableStreamType[Attribute]


class ReadOperation(IRDLOperation, abc.ABC):
    """
    Abstract base class for operations that read from a stream.
    """

    T: ClassVar[VarConstraint[Attribute]] = VarConstraint("T", AnyAttr())

    stream = operand_def(ReadableStreamType[Attribute].constr(element_type=T))
    res = result_def(T)

    def __init__(self, stream: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream_type := stream.type, ReadableStreamType)
            stream_type = cast(ReadableStreamType[Attribute], stream_type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_characters("from")
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved = parser.resolve_operand(unresolved, ReadableStreamType(result_type))
        return cls(resolved, result_type)

    def print(self, printer: Printer):
        printer.print_string(" from ")
        printer.print(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


class WriteOperation(IRDLOperation, abc.ABC):
    """
    Abstract base class for operations that write to a stream.
    """

    T: ClassVar[VarConstraint[Attribute]] = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    stream = operand_def(WritableStreamType[Attribute].constr(element_type=T))

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        unresolved_value = parser.parse_unresolved_operand()
        parser.parse_characters("to")
        unresolved_stream = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved_value = parser.resolve_operand(unresolved_value, result_type)
        resolved_stream = parser.resolve_operand(
            unresolved_stream, WritableStreamType(result_type)
        )
        return cls(resolved_value, resolved_stream)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(" to ")
        printer.print_ssa_value(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)


@irdl_op_definition
class ReadOp(ReadOperation):
    name = "stream.read"


@irdl_op_definition
class WriteOp(WriteOperation):
    name = "stream.write"


Stream = Dialect(
    "stream",
    [
        ReadOp,
        WriteOp,
    ],
    [
        ReadableStreamType,
        WritableStreamType,
    ],
)
