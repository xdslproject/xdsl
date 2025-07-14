from __future__ import annotations

from typing import TYPE_CHECKING

from xdsl.ir import Data
from xdsl.utils.runtime_final import runtime_final

if TYPE_CHECKING:
    from xdsl.parser import AttrParser
    from xdsl.printer import Printer


@runtime_final
class IntAttr(Data[int]):
    name = "builtin.int"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            data = parser.parse_integer()
            return data

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(f"{self.data}")

    def __bool__(self) -> bool:
        """Returns True if value is non-zero."""
        return bool(self.data)
