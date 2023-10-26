from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
)
from xdsl.dialects.stream import (
    ReadableStreamType,
    WritableStreamType,
)
from xdsl.ir import (
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


@irdl_attr_definition
class StridePatternType(ParametrizedAttribute, TypeAttribute):
    name = "stream.stride_pattern_type"


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "snitch_stream.generic"

    repeat_count = operand_def(riscv.IntRegisterType)
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
    name = "snitch_stream.read"

    T = Annotated[riscv.FloatRegisterType, ConstraintVar("T")]

    stream: Operand = operand_def(ReadableStreamType[T])
    res: OpResult = result_def(T)

    def __init__(self, stream: SSAValue, result_type: riscv.FloatRegisterType):
        super().__init__(operands=[stream], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser) -> ReadOp:
        unresolved = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        type_start_pos = parser.pos
        result_type = parser.parse_attribute()
        if not isinstance(result_type, riscv.FloatRegisterType):
            parser.raise_error("Expected a floating point register", type_start_pos)
        resolved = parser.resolve_operand(unresolved, ReadableStreamType(result_type))
        return ReadOp(resolved, result_type)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print(self.stream)
        printer.print_string(" : ")
        printer.print_attribute(self.res.type)


@irdl_op_definition
class WriteOp(IRDLOperation):
    name = "snitch_stream.write"

    T = Annotated[riscv.FloatRegisterType, ConstraintVar("T")]

    stream: Operand = operand_def(WritableStreamType[T])
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
            unresolved_stream, ReadableStreamType(result_type)
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
    name = "snitch_stream.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class StridePatternOp(IRDLOperation):
    """
    Specifies a stream access pattern reading from a pointer sequentially.
    """

    name = "snitch_stream.stride_pattern"

    stream = result_def(StridePatternType)
    ub = attr_def(ArrayAttr[IntAttr])
    strides = attr_def(ArrayAttr[IntAttr])
    dm = attr_def(IntAttr)

    def __init__(
        self,
        ub: ArrayAttr[IntAttr],
        strides: ArrayAttr[IntAttr],
        dm: IntAttr,
    ):
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

    name = "snitch_stream.strided_read"

    pointer = operand_def(riscv.IntRegisterType)
    pattern = operand_def(StridePatternType)
    stream = result_def(ReadableStreamType[riscv.FloatRegisterType])
    dm = attr_def(IntAttr)
    rank = attr_def(IntAttr)

    def __init__(
        self,
        pointer: SSAValue,
        pattern: SSAValue,
        register: riscv.FloatRegisterType,
        dm: IntAttr,
        rank: IntAttr,
    ):
        super().__init__(
            operands=[pointer, pattern],
            result_types=[ReadableStreamType(register)],
            attributes={
                "dm": dm,
                "rank": rank,
            },
        )


@irdl_op_definition
class StridedWriteOp(IRDLOperation):
    """
    Generates a stream writing to a pointer sequentially.
    """

    name = "snitch_stream.strided_write"

    pointer = operand_def(riscv.IntRegisterType)
    pattern = operand_def(StridePatternType)
    stream = result_def(WritableStreamType[riscv.FloatRegisterType])
    dm = attr_def(IntAttr)
    rank = attr_def(IntAttr)

    def __init__(
        self,
        pointer: SSAValue,
        pattern: SSAValue,
        register: riscv.FloatRegisterType,
        dm: IntAttr,
        rank: IntAttr,
    ):
        super().__init__(
            operands=[pointer, pattern],
            result_types=[WritableStreamType(register)],
            attributes={
                "dm": dm,
                "rank": rank,
            },
        )


SnitchStream = Dialect(
    [
        GenericOp,
        YieldOp,
        ReadOp,
        WriteOp,
        StridedReadOp,
        StridedWriteOp,
        StridePatternOp,
    ],
)
