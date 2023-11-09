from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import Self

from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
)
from xdsl.dialects.stream import (
    ReadableStreamType,
    WritableStreamType,
)
from xdsl.dialects.utils import parse_return_op_like, print_return_op_like
from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
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
    name = "snitch_stream.stride_pattern_type"


@irdl_op_definition
class GenericOp(IRDLOperation):
    """
    An operation that repeatedly reads from the input streams, performs the calculation in
    the `body`, and writes the results to the output streams. The number of times that
    this operation repeats is specified by `repeat_count`.
    """

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
class YieldOp(IRDLOperation):
    name = "snitch_stream.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[operands])

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, self.values)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs, args = parse_return_op_like(parser)
        op = cls(*args)
        op.attributes.update(attrs)
        return op


@irdl_op_definition
class StridePatternOp(IRDLOperation):
    """
    Specifies a stream access pattern reading from or writing to a pointer.
    `ub` specifies the upper bounds of the iteration variables.
    `strides` specifies the strides in bytes of the iteration variables.

    For example, to read sequentially the elements of a 2x3xf32 matrix in row-major order:
    `ub = [2,3], strides = [12, 4]`

    The index for each iteration will be calculated like this:
    (0, 0) -> 0*12 + 0*4 = 0
    (0, 1) -> 0*12 + 1*4 = 4
    (0, 2) -> 0*12 + 2*4 = 8
    (1, 0) -> 1*12 + 0*4 = 12
    (1, 1) -> 1*12 + 1*4 = 16
    (1, 2) -> 1*12 + 2*4 = 18
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
    Generates a stream reading from a pointer according to the provided pattern.
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
    Generates a stream writing to a pointer according to the provided pattern.
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
    "snitch-stream",
    [
        GenericOp,
        YieldOp,
        StridedReadOp,
        StridedWriteOp,
        StridePatternOp,
    ],
    [
        StridePatternType,
    ],
)
