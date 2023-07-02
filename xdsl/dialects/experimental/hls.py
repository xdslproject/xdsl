from __future__ import annotations


from xdsl.dialects.builtin import (
    Attribute,
    StringAttr,
)
from xdsl.ir import Operation, SSAValue, Dialect
from xdsl.irdl import (
    irdl_op_definition,
    Operand,
    IRDLOperation,
    operand_def,
    opt_attr_def,
)
from xdsl.dialects.builtin import i32


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


HLS = Dialect([PragmaPipeline, PragmaUnroll, PragmaDataflow, PragmaArrayPartition], [])
