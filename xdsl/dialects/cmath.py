from __future__ import annotations

from typing import Union

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
    VerifyException,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


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
        if not isinstance(self.op.typ, ComplexType):
            raise VerifyException("Expected complex type")
        if self.op.typ.data != self.res.typ:
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
        if self.lhs.typ != self.rhs.typ and self.rhs.typ != self.result.typ:
            raise VerifyException("expect all input and output types to be equal")

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Mul:
        operand1 = SSAValue.get(operand1)
        return Mul.build(operands=[operand1, operand2], result_types=[operand1.typ])


CMath = Dialect([Norm, Mul], [ComplexType])
