from contextlib import contextmanager
from dataclasses import dataclass

from xdsl.utils.base_printer import BasePrinter
from xdsl.utils.colors import RESET, Colors
from xdsl.utils.lexer import ContentT


@dataclass(eq=False, repr=False, init=False)
class ColorPrinter(BasePrinter[ContentT]):
    """
    A printer for printing colored text to a terminal window.
    Colors are set by calling the `colored` context manager.
    """

    _in_color_block: bool = False

    @contextmanager
    def colored(self, color: Colors | None):
        """
        No-op if given color is None or the printer is already in a color block.
        Otherwise, anything printed in this context manager will be printed in
        the given color.

        When nesting this context manager, the color is determined by the outermost
        call. For example:
        ```python
        with printer.colored(Colors.BLUE):
            with printer.colored(Colors.RED):
                printer.print_string("test")
        ```
        will print "test" in blue, and not red.
        """

        if self._in_color_block or color is None:
            yield
        else:
            self._in_color_block = True
            self._write(self._encode(str(color)))
            yield
            self._write(self._encode(str(RESET)))
            self._in_color_block = False
