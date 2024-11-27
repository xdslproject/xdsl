from __future__ import annotations

from fractions import Fraction

from xdsl.ir import Data, Dialect, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class AngleAttr(Data[Fraction], TypeAttribute):
    """
    Attribute that wraps around a fraction, implicitly multiplying by pi and keeping the result in the range [0,2pi)
    """

    name = "quantum.angle"

    @classmethod
    def from_fraction(cls, numerator: int, denominator: int = 1) -> AngleAttr:
        return AngleAttr(Fraction(numerator, denominator) % 2)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> Fraction:
        with parser.in_angle_brackets():
            negate = -1 if parser.parse_optional_punctuation("-") else 1

            numerator = parser.parse_optional_integer()
            if numerator is None:
                numerator = 1
            numerator = numerator * negate

            if numerator == 0:
                return Fraction(0, 1)

            parser.parse_characters("pi")
            denominator = (
                parser.parse_integer() if parser.parse_optional_punctuation(":") else 1
            )
            return Fraction(numerator, denominator) % 2

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if self.data == 0:
                printer.print(0)
                return

            if self.data.numerator != 1:
                printer.print(self.data.numerator)

            printer.print("pi")
            if self.data.denominator != 1:
                printer.print(":", self.data.denominator)

    def __add__(self, other: AngleAttr) -> AngleAttr:
        return AngleAttr.new((self.data + other.data) % 2)

    def __sub__(self, other: AngleAttr) -> AngleAttr:
        return AngleAttr.new((self.data - other.data) % 2)

    def __neg__(self) -> AngleAttr:
        return AngleAttr.new(-self.data % 2)


QUANTUM = Dialect(
    "quantum",
    [],
    [AngleAttr],
)
