from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Union

from xdsl.dialects.builtin import (ContainerOf, Float16Type, Float64Type, IndexType,
                                   IntegerType, Float32Type)
from xdsl.ir import Dialect, MLContext, OpResult, Operation, SSAValue
from xdsl.irdl import AnyOf, Operand, irdl_op_definition
from xdsl.utils.exceptions import VerifyException

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))

@dataclass
class Math:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(IPowI)

@dataclass
class BinaryOperation(Operation):
    """A generic operation. Operation definitions inherit this class."""

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException(
                "expect all input and result types to be equal")

    def __hash__(self) -> int:
        return id(self)


@irdl_op_definition
class IPowI(BinaryOperation):
    name: str = "math.ipowi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]) -> IPowI:
        operand1 = SSAValue.get(operand1)
        return IPowI.build(operands=[operand1, operand2],
                           result_types=[operand1.typ])

Math = Dialect([IPowI], [])
