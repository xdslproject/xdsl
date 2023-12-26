from __future__ import annotations

from abc import ABC
from typing import TypeAlias, TypeVar

from xdsl.dialects.builtin import AnyTensorType, SSAValue
from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
)
from xdsl.irdl import (
    Attribute,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer

ArgT = TypeVar("ArgT", bound=Attribute)
Operand: TypeAlias = SSAValue


class ElementwiseBinOpBase(IRDLOperation, ABC):
    """Base class for element-wise binary operations on tensors with Numpy-style broadcasting."""

    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    res: OpResult = result_def(AnyTensorType)

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[[]],
        )

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_unresolved_operand()
        parser.parse_characters(",")
        rhs = parser.parse_unresolved_operand()
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        type = parser.parse_type()
        operands = parser.resolve_operands([lhs, rhs], [type, type], parser.pos)
        return cls(operands[0], operands[1], attributes)

    def print(self, printer: Printer) -> None:
        printer.print("(", self.lhs, ", ", self.rhs, ") ")
        printer.print_op_attributes(self.attributes)
        for x in [
            " : ",
            " (",
            self.lhs.type,
            ", ",
            self.rhs.type,
            ") -> ",
            self.res.type,
        ]:
            printer.print(x)


@irdl_op_definition
class Add(ElementwiseBinOpBase):
    name = "onnx.Add"


@irdl_op_definition
class Sub(ElementwiseBinOpBase):
    name = "onnx.Sub"


@irdl_op_definition
class Mul(ElementwiseBinOpBase):
    name = "onnx.Mul"


@irdl_op_definition
class Div(ElementwiseBinOpBase):
    name = "onnx.Div"


ONNX = Dialect("onnx", [Add, Sub, Mul, Div])
