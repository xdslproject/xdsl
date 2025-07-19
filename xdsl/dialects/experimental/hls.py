from __future__ import annotations

from xdsl.dialects.builtin import (
    Attribute,
    DenseArrayBase,
    IntegerType,
    ParametrizedAttribute,
    StringAttr,
    i64,
)
from xdsl.ir import Dialect, Operation, Region, SSAValue, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class HLSYieldOp(IRDLOperation):
    name = "hls.yield"
    arguments = var_operand_def()

    traits = traits_def(IsTerminator())

    @staticmethod
    def get(*operands: SSAValue | Operation) -> HLSYieldOp:
        return HLSYieldOp.create(
            operands=[SSAValue.get(operand) for operand in operands]
        )


@irdl_op_definition
class PragmaPipelineOp(IRDLOperation):
    name = "hls.pipeline"
    ii = operand_def(IntegerType)

    def __init__(self, ii: SSAValue | Operation):
        super().__init__(operands=[ii])


@irdl_op_definition
class PragmaUnrollOp(IRDLOperation):
    name = "hls.unroll"
    factor = operand_def(IntegerType)

    def __init__(self, factor: SSAValue | Operation):
        super().__init__(operands=[factor])


@irdl_op_definition
class PragmaDataflowOp(IRDLOperation):
    name = "hls.dataflow"

    body = region_def()

    def __init__(self, region: Region):
        super().__init__(regions=[region])


@irdl_op_definition
class PragmaArrayPartitionOp(IRDLOperation):
    name = "hls.array_partition"
    variable = opt_attr_def(StringAttr)
    array_type = opt_attr_def()  # look at memref.Global
    factor = operand_def()
    dim = operand_def()

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

    element_type: Attribute


@irdl_op_definition
class HLSStreamOp(IRDLOperation):
    name = "hls.stream"
    elem_type = attr_def()
    result = result_def(HLSStreamType)  # This should be changed to HLSStreamType

    @staticmethod
    def get(elem_type: Attribute) -> HLSStreamOp:
        attrs: dict[str, Attribute] = {}

        attrs["elem_type"] = elem_type

        stream_type = HLSStreamType(elem_type)
        return HLSStreamOp.build(result_types=[stream_type], attributes=attrs)


@irdl_op_definition
class HLSStreamWriteOp(IRDLOperation):
    name = "hls.write"
    element = operand_def()
    stream = operand_def(HLSStreamType)

    def __init__(self, element: SSAValue | Operation, stream: SSAValue | Operation):
        super().__init__(operands=[element, stream])


@irdl_op_definition
class HLSStreamReadOp(IRDLOperation):
    name = "hls.read"
    stream = operand_def(HLSStreamType)
    res = result_def()

    def __init__(self, stream: SSAValue):
        assert isinstance(stream.type, HLSStreamType)
        print("TYPE STREAM: ", type(stream.type))
        super().__init__(operands=[stream], result_types=[stream.type.element_type])


@irdl_op_definition
class HLSExtractStencilValueOp(IRDLOperation):
    name = "hls.extract_stencil_value"

    position = attr_def(DenseArrayBase.constr(i64))
    container = operand_def(Attribute)

    res = result_def(Attribute)

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
        PragmaPipelineOp,
        PragmaUnrollOp,
        PragmaDataflowOp,
        PragmaArrayPartitionOp,
        HLSStreamOp,
        HLSStreamWriteOp,
        HLSStreamReadOp,
        HLSYieldOp,
        HLSExtractStencilValueOp,
    ],
    [HLSStreamType],
)
