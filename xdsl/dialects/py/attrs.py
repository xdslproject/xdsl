from xdsl.ir import (
    Data,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    BaseAttr,
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


class Object:
    def __init__(self, value: object):
        self.value = value
        self.name = value.__repr__()


@irdl_attr_definition
class ConstantValue(Data[Object]):
    name = "py.const"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.__str__())

    @staticmethod
    def constr() -> AttrConstraint:
        return BaseAttr(ObjectType)
