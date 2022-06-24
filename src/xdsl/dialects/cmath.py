
from __future__ import annotations
from dataclasses import dataclass

from xdsl.dialects.builtin import * 

from xdsl.irdl import *
from xdsl.ir import *
from xdsl.dialects.irdl import *
from xdsl.dialects.func import *
from xdsl.dialects.arith import *
from xdsl.dialects.builtin import *

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.parser import Parser
    from xdsl.printer import Printer

@dataclass
class Cmath:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(ComplexType)

        self.ctx.register_op(Norm)
        self.ctx.register_op(Mul)

@irdl_attr_definition
class ComplexType(ParametrizedAttribute):
    name = "complex"
    data: ParameterDef[Float64Type | Float32Type]

@irdl_op_definition
class Norm(Operation):
    name: str = "irdl.norm"
    
    operands = OperandDef(ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])] ) )
    results = ResultDef(AnyOf([Float32Type, Float64Type]))

    # TODO replace with trait
    def verify_(self) -> None:
        if self.operands.typ != self.results.typ:
            raise VerifyException(
                "expect all input and output types to be equal")

@irdl_op_definition
class Mul(Operation):
    name: str = "irdl.mul"
    data: ParameterDef[Float64Type | Float32Type]
    
    lhs = OperandDef(ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])] ) )
    rhs = OperandDef(ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])] ) )
    results = ResultDef(ParamAttrConstraint(ComplexType, [AnyOf([Float32Type, Float64Type])]))

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs != self.rhs.typ and self.rhs.typ != self.results.typ:
            raise VerifyException(
                "expect all input and output types to be equal")


