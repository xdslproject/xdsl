from __future__ import annotations

from xdsl.dialects.builtin import (
    Attribute,
    DenseArrayBase,
    IntegerType,
    ParametrizedAttribute,
    StringAttr,
)
from xdsl.ir import Dialect, Operation, OpResult, Region, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class HLSYield(IRDLOperation):
    name = "hls.yield"
    arguments: VarOperand = var_operand_def(AnyAttr())

    traits = frozenset([IsTerminator()])

    @staticmethod
    def get(*operands: SSAValue | Operation) -> HLSYield:
        return HLSYield.create(operands=[SSAValue.get(operand) for operand in operands])


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

    body: Region = region_def()

    def __init__(self, region: Region):
        super().__init__(regions=[region])


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
    element: Operand = operand_def(AnyAttr())
    stream: Operand = operand_def(HLSStreamType)

    def __init__(self, element: SSAValue | Operation, stream: SSAValue | Operation):
        super().__init__(operands=[element, stream])


@irdl_op_definition
class HLSStreamRead(IRDLOperation):
    name = "hls.read"
    stream: Operand = operand_def(HLSStreamType)
    res: OpResult = result_def(AnyAttr())

    def __init__(self, stream: SSAValue):
        assert isinstance(stream.type, HLSStreamType)
        print("TYPE STREAM: ", type(stream.type))
        super().__init__(operands=[stream], result_types=[stream.type.element_type])


@irdl_op_definition
class HLSExtractStencilValue(IRDLOperation):
    name = "hls.extract_stencil_value"

    position: DenseArrayBase = attr_def(DenseArrayBase)
    container: Operand = operand_def(Attribute)

    res: OpResult = result_def(Attribute)

    def __init__(
        self,
        position: DenseArrayBase,
        container: SSAValue | Operation,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[container],
            attributes={
                "position": position,
            },
            result_types=[result_type],
        )


HLS = Dialect(
    "hls",
    [
        PragmaPipeline,
        PragmaUnroll,
        PragmaDataflow,
        PragmaArrayPartition,
        HLSStream,
        HLSStreamWrite,
        HLSStreamRead,
        HLSYield,
        HLSExtractStencilValue,
    ],
    [HLSStreamType],
)
