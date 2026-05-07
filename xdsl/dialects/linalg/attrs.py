from __future__ import annotations

from enum import auto

from xdsl.ir import EnumAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.str_enum import StrEnum


class IteratorType(StrEnum):
    "Iterator type for linalg trait"

    PARALLEL = auto()
    REDUCTION = auto()
    WINDOW = auto()


@irdl_attr_definition
class IteratorTypeAttr(EnumAttribute[IteratorType]):
    name = "linalg.iterator_type"

    @classmethod
    def parallel(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.PARALLEL)

    @classmethod
    def reduction(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.REDUCTION)

    @classmethod
    def window(cls) -> IteratorTypeAttr:
        return IteratorTypeAttr(IteratorType.WINDOW)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> IteratorType:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)
