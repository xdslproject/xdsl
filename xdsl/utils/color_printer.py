from contextlib import contextmanager
from dataclasses import dataclass, field

from xdsl.utils.base_printer import BasePrinter
from xdsl.utils.colors import RESET, Colors


@dataclass(eq=False, repr=False)
class ColorPrinter(BasePrinter):
    """
    A printer for printing colored text to a terminal window.
    """

    _in_color_block: bool = field(default=False, init=False)

    @contextmanager
    def colored(self, color: Colors | None):
        if self._in_color_block or color is None:
            yield
        else:
            self._in_color_block = True
            print(color, end="", file=self.stream)
            yield
            print(RESET, end="", file=self.stream)
            self._in_color_block = False
