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
from xdsl.parser import Parser
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
class StridePatternType(ParametrizedAttribute, TypeAttribute):
    name = "stream.stride_pattern_type"


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "stream.generic"

    repeat_count = operand_def(IndexType)
    inputs = var_operand_def(ReadableStreamType)
    outputs = var_operand_def(WritableStreamType)

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        repeat_count: SSAValue,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
    ) -> None:
        super().__init__(
            operands=[repeat_count, inputs, outputs],
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
        super().__init__(
            result_types=[StridePatternType()],
            attributes={
                "ub": ub,
                "strides": strides,
                "dm": dm,
            },
        )


@irdl_op_definition
class StridedReadOp(IRDLOperation):
    """
    Generates a stream reading from a memref sequentially.
    """

    name = "stream.strided_read"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    pattern = operand_def(StridePatternType())
    stream = result_def(ReadableStreamType[T])
    dm = attr_def(IntAttr)
    rank = attr_def(IntAttr)

    def __init__(
        self,
        memref: SSAValue,
        pattern: SSAValue,
        dm: IntAttr,
        rank: IntAttr,
    ):
        assert isinstance(memref.type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref.type)
        super().__init__(
            operands=[memref, pattern],
            result_types=[ReadableStreamType(memref_type.element_type)],
            attributes={
                "dm": dm,
                "rank": rank,
            },
        )


@irdl_op_definition
class StridedWriteOp(IRDLOperation):
    """
    Generates a stream writing from a memref sequentially.
    """

    name = "stream.strided_write"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    pattern = operand_def(StridePatternType())
    stream = result_def(WritableStreamType[T])
    dm = attr_def(IntAttr)
    rank = attr_def(IntAttr)

    def __init__(self, memref: SSAValue, pattern: SSAValue, dm: IntAttr, rank: IntAttr):
        assert isinstance(memref.type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref.type)
        super().__init__(
            operands=[memref, pattern],
            result_types=[WritableStreamType(memref_type.element_type)],
            attributes={
                "dm": dm,
                "rank": rank,
            },
        )


Stream = Dialect(
    "stream",
    [
        GenericOp,
        YieldOp,
        ReadOp,
        WriteOp,
        StridedReadOp,
        StridedWriteOp,
        StridePatternOp,
    ],
    [
        ReadableStreamType,
        WritableStreamType,
        StridePatternType,
    ],
)
