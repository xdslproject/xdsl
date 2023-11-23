from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Generic, TypeVar, cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerType,
    IndexType,
    IntAttr,
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator

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


@irdl_attr_definition
class ReadableStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "stream.readable"


@irdl_attr_definition
class WritableStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "stream.writable"


@irdl_attr_definition
class StridePatternType(Data[int], TypeAttribute):
    name = "stream.stride_pattern_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            return parser.parse_integer()

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "stream.generic"

    T = Annotated[Attribute, ConstraintVar("T")]

    repeat_count = operand_def(IndexType)
    inputs = var_operand_def(MemRefType[T] | T)
    outputs = var_operand_def(MemRefType[T])
    stride_patterns = var_operand_def(StridePatternType)

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        repeat_count: SSAValue,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        stride_patterns: Sequence[SSAValue],
        body: Region,
    ) -> None:
        super().__init__(
            operands=[repeat_count, inputs, outputs, stride_patterns],
            regions=[body],
        )


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "stream.read"

    T = Annotated[Attribute, ConstraintVar("T")]

    stream: Operand = operand_def(ReadableStreamType[T])
    res: OpResult = result_def(T)

    def __init__(self, stream: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream_type := stream.type, ReadableStreamType)
            stream_type = cast(ReadableStreamType[Attribute], stream_type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> ReadOp:
        parser.parse_characters("from")
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved = parser.resolve_operand(unresolved, ReadableStreamType(result_type))
        return ReadOp(resolved, result_type)

    def print(self, printer: Printer):
        printer.print_string(" from ")
        printer.print(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class WriteOp(IRDLOperation):
    name = "stream.write"

    T = Annotated[Attribute, ConstraintVar("T")]

    value: Operand = operand_def(T)
    stream: Operand = operand_def(WritableStreamType[T])

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[value, stream])

    @classmethod
    def parse(cls, parser: Parser) -> WriteOp:
        unresolved_value = parser.parse_unresolved_operand()
        parser.parse_characters("to")
        unresolved_stream = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved_value = parser.resolve_operand(unresolved_value, result_type)
        resolved_stream = parser.resolve_operand(
            unresolved_stream, WritableStreamType(result_type)
        )
        return WriteOp(resolved_value, resolved_stream)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.value)
        printer.print_string(" to ")
        printer.print_ssa_value(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "stream.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class StridePatternOp(IRDLOperation):
    """
    Specifies a stream access pattern reading from a memref sequentially.
    """

    name = "stream.stride_pattern"

    pattern = result_def(StridePatternType)
    ub = attr_def(ArrayAttr[IntAttr])
    strides = attr_def(ArrayAttr[IntAttr])
    dm = attr_def(IntAttr)

    def __init__(
        self,
        ub: ArrayAttr[IntAttr],
        strides: ArrayAttr[IntAttr],
        dm: IntAttr,
    ) -> None:
        rank = len(ub.data)
        assert rank == len(strides.data)
        super().__init__(
            result_types=[StridePatternType(rank)],
            attributes={
                "ub": ub,
                "strides": strides,
                "dm": dm,
            },
        )


Stream = Dialect(
    "stream",
    [
        GenericOp,
        YieldOp,
        ReadOp,
        WriteOp,
        StridePatternOp,
    ],
    [
        ReadableStreamType,
        WritableStreamType,
        StridePatternType,
    ],
)
