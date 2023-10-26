from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
)
from xdsl.dialects.stream import (
    ReadableStreamType,
    StridePatternType,
    WritableStreamType,
)
from xdsl.ir import (
    Dialect,
    Operation,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    VarOperand,
    attr_def,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


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
        StridedReadOp,
        StridedWriteOp,
        StridePatternOp,
    ],
)
