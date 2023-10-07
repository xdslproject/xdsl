from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Generic, TypeVar, cast

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ContainerType,
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
    prop_def,
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
class InputStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "input_stream"


@irdl_attr_definition
class OutputStreamType(Generic[_StreamTypeElement], StreamType[_StreamTypeElement]):
    name = "output_stream"


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "stream.generic"

    inputs = var_operand_def(InputStreamType)
    outputs = var_operand_def(OutputStreamType)

    body = region_def("single_block")

    static_loop_ranges = prop_def(ArrayAttr[IntAttr])

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        static_loop_ranges: ArrayAttr[IntAttr],
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            properties={
                "static_loop_ranges": static_loop_ranges,
            },
            regions=[body],
        )


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "stream.read"

    T = Annotated[Attribute, ConstraintVar("T")]

    stream: Operand = operand_def(InputStreamType[T])
    res: OpResult = result_def(T)

    def __init__(self, stream: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            assert isinstance(stream.type, InputStreamType)
            stream_type = cast(InputStreamType[Attribute], stream.type)
            result_type = stream_type.element_type
        super().__init__(operands=[stream], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> ReadOp:
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved = parser.resolve_operand(unresolved, InputStreamType(result_type))
        return ReadOp(resolved, result_type)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class WriteOp(IRDLOperation):
    name = "stream.write"

    T = Annotated[Attribute, ConstraintVar("T")]

    stream: Operand = operand_def(OutputStreamType[T])
    value: Operand = operand_def(T)

    def __init__(self, value: SSAValue, stream: SSAValue):
        super().__init__(operands=[stream, value])

    @classmethod
    def parse(cls, parser: Parser) -> WriteOp:
        unresolved_stream = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        unresolved_value = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        resolved_value = parser.resolve_operand(unresolved_value, result_type)
        resolved_stream = parser.resolve_operand(
            unresolved_stream, InputStreamType(result_type)
        )
        return WriteOp(resolved_value, resolved_stream)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.stream)
        printer.print_string(", ")
        printer.print_ssa_value(self.value)
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
class StridedReadOp(IRDLOperation):
    """
    Generates a stream reading from a memref sequentially.
    """

    name = "stream.strided_read"

    T = Annotated[Attribute, ConstraintVar("T")]

    memref = operand_def(MemRefType[T])
    stream = result_def(InputStreamType[T])
    ub = attr_def(ArrayAttr[IntAttr])
    indexing_map = attr_def(AffineMapAttr)

    def __init__(
        self, memref: SSAValue, ub: ArrayAttr[IntAttr], indexing_map: AffineMapAttr
    ):
        assert isinstance(memref.type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref.type)
        super().__init__(
            operands=[memref],
            result_types=[InputStreamType(memref_type.element_type)],
            attributes={
                "ub": ub,
                "indexing_map": indexing_map,
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
    stream = result_def(OutputStreamType[T])
    ub = attr_def(ArrayAttr[IntAttr])
    indexing_map = attr_def(AffineMapAttr)

    def __init__(
        self, memref: SSAValue, ub: ArrayAttr[IntAttr], indexing_map: AffineMapAttr
    ):
        assert isinstance(memref.type, MemRefType)
        memref_type = cast(MemRefType[Attribute], memref.type)
        super().__init__(
            operands=[memref],
            result_types=[OutputStreamType(memref_type.element_type)],
            attributes={
                "ub": ub,
                "indexing_map": indexing_map,
            },
        )


Linalg = Dialect(
    [
        GenericOp,
        YieldOp,
        ReadOp,
        WriteOp,
        StridedReadOp,
        StridedWriteOp,
    ],
    [
        InputStreamType,
        OutputStreamType,
    ],
)
