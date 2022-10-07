from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated

from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.ir import MLContext, MLIRType, OpResult, SSAValue
from xdsl.irdl import (S_OperandDef, S_ResultDef, irdl_op_definition,
                       Operation, OperandDef, irdl_attr_definition,
                       ParameterDef, ParamAttrConstraint, AnyOf, ResultDef,
                       ParametrizedAttribute, VerifyException)


@dataclass
class CMath:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(ComplexType)

        self.ctx.register_op(Norm)
        self.ctx.register_op(Mul)


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, MLIRType):
    name = "cmath.complex"
    data: ParameterDef[Float64Type | Float32Type]


@irdl_op_definition
class Norm(Operation):
    name: str = "cmath.norm"

    op: S_OperandDef[Annotated[
        SSAValue,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]]
    res: S_ResultDef[Annotated[OpResult, Float32Type | Float64Type]]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.op.typ.data != self.res.typ:
            raise VerifyException(
                "expect all input and output types to be equal")


@irdl_op_definition
class Mul(Operation):
    name: str = "cmath.mul"

    lhs: S_OperandDef[Annotated[
        SSAValue,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]]
    rhs: S_OperandDef[Annotated[
        SSAValue,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]]
    res: S_ResultDef[Annotated[
        OpResult,
        ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])])]]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs != self.rhs.typ and self.rhs.typ != self.res.typ:
            raise VerifyException(
                "expect all input and output types to be equal")
