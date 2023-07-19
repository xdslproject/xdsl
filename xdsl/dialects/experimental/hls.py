from __future__ import annotations

from xdsl.dialects.builtin import Attribute, ParametrizedAttribute, StringAttr
from xdsl.ir import Operation, SSAValue, Dialect, TypeAttribute, OpResult
from xdsl.irdl import (
    AnyAttr,
    irdl_op_definition,
    irdl_attr_definition,
    Operand,
    IRDLOperation,
    operand_def,
    opt_attr_def,
    attr_def,
    result_def,
    ParameterDef,
)

from typing import TypeVar

from xdsl.dialects.builtin import IntegerType

_StreamTypeElement = TypeVar("_StreamTypeElement", bound=Attribute, covariant=True)


@irdl_op_definition
class PragmaPipeline(IRDLOperation):
    name = "hls.pipeline"
    ii: Operand = operand_def(IntegerType)

    def __init__(self, ii: SSAValue | Operation):
        super().__init__(operands=[ii])


@irdl_op_definition
class PragmaUnroll(IRDLOperation):
    name = "hls.unroll"
    factor: Operand = operand_def(IntegerType)

    def __init__(self, factor: SSAValue | Operation):
        super().__init__(operands=[factor])


@irdl_op_definition
class PragmaDataflow(IRDLOperation):
    name = "hls.dataflow"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class PragmaArrayPartition(IRDLOperation):
    name = "hls.array_partition"
    variable: StringAttr | None = opt_attr_def(StringAttr)
    array_type: Attribute | None = opt_attr_def(Attribute)  # look at memref.Global
    factor: Operand = operand_def()
    dim: Operand = operand_def()

    def __init__(
        self,
        variable: StringAttr,
        array_type: Attribute,
        factor: SSAValue | Operation,
        dim: SSAValue | Operation,
    ):
        super().__init__(
            operands=[factor, dim],
            attributes={"variable": variable, "array_type": array_type},
        )


@irdl_attr_definition
class HLSStreamType(ParametrizedAttribute, TypeAttribute):
    name = "hls.streamtype"

    element_type: ParameterDef[Attribute]

    @staticmethod
    def get(element_type: Attribute):
        return HLSStreamType([element_type])


@irdl_op_definition
class HLSStream(IRDLOperation):
    name = "hls.stream"
    elem_type: Attribute = attr_def(Attribute)
    result: OpResult = result_def(
        HLSStreamType
    )  # This should be changed to HLSStreamType

    @staticmethod
    def get(elem_type: Attribute) -> HLSStream:
        attrs: dict[str, Attribute] = {}

        attrs["elem_type"] = elem_type

        stream_type = HLSStreamType([elem_type])
        return HLSStream.build(result_types=[stream_type], attributes=attrs)


@irdl_op_definition
class HLSStreamWrite(IRDLOperation):
    name = "hls.write"
    element: Operand = operand_def()
    stream: Operand = operand_def(HLSStreamType)

    def __init__(self, element: Operation, stream: HLSStreamType):
        super().__init__(operands=[element, stream])


@irdl_op_definition
class HLSStreamRead(IRDLOperation):
    name = "hls.read"
    stream: Operand = operand_def(HLSStreamType)
    res: OpResult = result_def(AnyAttr())

    def __init__(self, stream: SSAValue | Operation):
        super().__init__(operands=[stream], result_types=[stream.typ.element_type])


HLS = Dialect(
    [
        PragmaPipeline,
        PragmaUnroll,
        PragmaDataflow,
        PragmaArrayPartition,
        HLSStream,
        HLSStreamWrite,
        HLSStreamRead,
    ],
    [HLSStreamType],
)
