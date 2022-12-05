from __future__ import annotations
from dataclasses import dataclass

from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.ir import MLIRType, ParametrizedAttribute, Operation, Dialect
from xdsl.irdl import (irdl_op_definition, irdl_attr_definition, OperandDef,
                       ParameterDef, ParamAttrConstraint, AnyOf, ResultDef,
                       VerifyException)


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, MLIRType):
    name = "cmath.complex"
    data: ParameterDef[Float64Type | Float32Type]


@irdl_op_definition
class Norm(Operation):
    name: str = "cmath.norm"

    op = OperandDef(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])]))
    res = ResultDef(AnyOf([Float32Type, Float64Type]))

    # TODO replace with trait
    def verify_(self) -> None:
        if self.op.typ.data != self.res.typ:
            raise VerifyException(
                "expect all input and output types to be equal")


@irdl_op_definition
class Mul(Operation):
    name: str = "cmath.mul"

    lhs = OperandDef(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])]))
    rhs = OperandDef(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])]))
    res = ResultDef(
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])]))

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs != self.rhs.typ and self.rhs.typ != self.res.typ:
            raise VerifyException(
                "expect all input and output types to be equal")


CMath = Dialect([Norm, Mul], [ComplexType])
