from __future__ import annotations
from typing import Annotated, Union

from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.ir import (MLIRType, ParametrizedAttribute, Operation, Dialect,
                     OpResult, SSAValue)
from xdsl.irdl import (irdl_op_definition, irdl_attr_definition, Operand,
                       ParameterDef, ParamAttrConstraint, AnyOf,
                       VerifyException)


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, MLIRType):
    name = "cmath.complex"
    data: ParameterDef[Float64Type | Float32Type]


@irdl_op_definition
class Norm(Operation):
    name: str = "cmath.norm"

    op: Annotated[
        Operand,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]
    res: Annotated[OpResult, AnyOf([Float32Type, Float64Type])]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.op.typ.data != self.res.typ:
            raise VerifyException(
                "expect all input and output types to be equal")


@irdl_op_definition
class Mul(Operation):
    name: str = "cmath.mul"

    lhs: Annotated[
        Operand,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]
    rhs: Annotated[
        Operand,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]
    result: Annotated[
        OpResult,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs != self.rhs.typ and self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> Mul:
        operand1 = SSAValue.get(operand1)
        return Mul.build(operands=[operand1, operand2],
                         result_types=[operand1.typ])


CMath = Dialect([Norm, Mul], [ComplexType])
