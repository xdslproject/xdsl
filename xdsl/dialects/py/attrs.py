from typing import Any

from xdsl.ir import (
    Data,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    GenericData,
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class ObjectType(GenericData[str], TypeAttribute):
    name = "py.type"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data)

    @staticmethod
    def constr() -> AttrConstraint:
        return AnyAttr()


@irdl_attr_definition
class ConstantValue(Data[Any]):
    name = "py.const"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.__str__())

    @staticmethod
    def constr() -> AttrConstraint:
        return AnyAttr()
