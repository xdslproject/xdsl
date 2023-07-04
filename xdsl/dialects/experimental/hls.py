from __future__ import annotations


from xdsl.dialects.builtin import (
    Attribute,
    ParametrizedAttribute,
    StringAttr,
)
from xdsl.ir import Operation, SSAValue, Dialect, TypeAttribute
from xdsl.irdl import (
    irdl_op_definition,
    irdl_attr_definition,
    Operand,
    IRDLOperation,
    operand_def,
    opt_attr_def,
    attr_def,
    Generic,
    result_def,
    ParameterDef,
)

from typing import TypeVar

from xdsl.dialects.builtin import i32, f16, IntegerType

_StreamTypeElement = TypeVar("_StreamTypeElement", bound=Attribute, covariant=True)


@irdl_op_definition
class PragmaPipeline(IRDLOperation):
    name = "hls.pipeline"
    ii: Operand = operand_def(i32)

    @staticmethod
    def get(ii: SSAValue | Operation):
        return PragmaPipeline.build(operands=[ii])


@irdl_op_definition
class PragmaUnroll(IRDLOperation):
    name = "hls.unroll"
    factor: Operand = operand_def()

    @staticmethod
    def get(factor: SSAValue | Operation):
        return PragmaUnroll.build(operands=[factor])


@irdl_op_definition
class PragmaDataflow(IRDLOperation):
    name = "hls.dataflow"

    @staticmethod
    def get():
        return PragmaDataflow.build()


@irdl_op_definition
class PragmaArrayPartition(IRDLOperation):
    name = "hls.array_partition"
    variable: StringAttr | None = opt_attr_def(StringAttr)
    array_type: Attribute | None = opt_attr_def(Attribute)  # look at memref.Global
    factor: Operand = operand_def()
    dim: Operand = operand_def()

    @staticmethod
    def get(
        variable: StringAttr,
        array_type: Attribute,
        factor: SSAValue | Operation,
        dim: SSAValue | Operation,
    ):
        return PragmaArrayPartition.build(
            operands=[factor, dim],
            attributes={"variable": variable, "array_type": array_type},
        )


@irdl_attr_definition
class HLSStreamType(ParametrizedAttribute, TypeAttribute):
    name = "hls.streamtype"

    element_type: ParameterDef[Attribute]

    def get(element_type: Attribute):
        return HLSStreamType(element_type)


# @irdl_op_definition
# class HLSStream(IRDLOperation):
#    name = "hls.stream"
#
#    @staticmethod
#    def get():
#        return HLSStream.build()


@irdl_op_definition
class HLSStream(IRDLOperation):
    name = "hls.stream"
    elem_type: Attribute = attr_def(Attribute)
    result: OpResult = result_def()  # This should be changed to HLSStreamType

    @staticmethod
    def get(elem_type: Attribute) -> HLSStream:
        attrs: dict[str, Attribute] = {}

        attrs["elem_type"] = elem_type

        stream_type = HLSStreamType([elem_type])
        return HLSStream.build(result_types=[stream_type], attributes=attrs)


# @irdl_op_definition
# class HLSExternalLoadOp(IRDLOperation):
#    name = "hls.external_load"
#    field: Operand = operand_def(Attribute)
#    #result: OpResult = result_def(FieldType[Attribute] | memref.MemRefType[Attribute])
#    result: OpResult = result_def(HLSStreamType[Attribute])
#
#    @staticmethod
#    def get(
#            arg: SSAValue | Operation,
#            res_type : HLSStreamType[Attribute]
#    ):
#        return HLSExternalLoadOp.build(operands=[arg], result_types=[res_type])


HLS = Dialect([PragmaPipeline, PragmaUnroll, PragmaDataflow, PragmaArrayPartition], [])
