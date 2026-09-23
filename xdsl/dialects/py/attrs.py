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

    def get_type(self) -> str:
        return self.data

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


class PyObject:
    def __init__(self, value: object):
        self.value = value
        self.name = value.__repr__()

    def __str__(self):
        return self.value.__str__()


@irdl_attr_definition
class ConstantValue(Data[PyObject]):
    name = "py.const"

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data.__str__())

    @staticmethod
    def constr() -> AttrConstraint:
        return BaseAttr(ObjectType)
