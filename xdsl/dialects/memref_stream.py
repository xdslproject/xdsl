from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntAttr,
)
from xdsl.dialects.memref import MemRefType
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    Operation,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintVar,
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
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


@irdl_attr_definition
class StridePatternType(Data[int], TypeAttribute):
    name = "memref_stream.stride_pattern_type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            return parser.parse_integer()

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            printer.print_string(str(self.data))


@irdl_op_definition
class GenericOp(IRDLOperation):
    name = "memref_stream.generic"

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
class YieldOp(IRDLOperation):
    name = "memref_stream.yield"

    values: VarOperand = var_operand_def()

    traits = frozenset([IsTerminator()])

    def __init__(self, *operands: SSAValue | Operation) -> None:
        super().__init__(operands=[SSAValue.get(operand) for operand in operands])


@irdl_op_definition
class StridePatternOp(IRDLOperation):
    """
    Specifies a stream access pattern reading from a memref sequentially.
    """

    name = "memref_stream.stride_pattern"

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


MemrefStream = Dialect(
    "memref_stream",
    [
        GenericOp,
        YieldOp,
        StridePatternOp,
    ],
    [
        StridePatternType,
    ],
)
