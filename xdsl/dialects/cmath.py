from __future__ import annotations

from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.ir import (
    Dialect,
    Operation,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    Operand,
    ParamAttrConstraint,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, TypeAttribute):
    name = "cmath.complex"
    data: ParameterDef[Float64Type | Float32Type]


@irdl_op_definition
class Norm(IRDLOperation):
    name = "cmath.norm"

    op: Operand = operand_def(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])
    )
    res: OpResult = result_def(AnyOf([Float32Type, Float64Type]))

    # TODO replace with trait
    def verify_(self) -> None:
        if not isinstance(self.op.type, ComplexType):
            raise VerifyException("Expected complex type")
        if self.op.type.data != self.res.type:
            raise VerifyException("expect all input and output types to be equal")


@irdl_op_definition
class Mul(IRDLOperation):
    name = "cmath.mul"

    lhs: Operand = operand_def(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])
    )
    rhs: Operand = operand_def(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])
    )
    result: OpResult = result_def(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])
    )

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.type != self.rhs.type and self.rhs.type != self.result.type:
            raise VerifyException("expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Operation | SSAValue, operand2: Operation | SSAValue) -> Mul:
        operand1 = SSAValue.get(operand1)
        return Mul.build(operands=[operand1, operand2], result_types=[operand1.type])


CMath = Dialect([Norm, Mul], [ComplexType])
