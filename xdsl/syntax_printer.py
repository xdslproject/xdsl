from xdsl.ir import SSAValue
from xdsl.printer import Printer
from xdsl.utils.color_printer import ColorPrinter
from xdsl.utils.colors import Colors


class SyntaxPrinter(Printer, ColorPrinter):
    """
    A printer for printing syntax-highlighted mlir code to a terminal.
    """

    def print_ssa_value(self, value: SSAValue) -> str:
        with self.colored(Colors.BRIGHT_MAGENTA):
            return super().print_ssa_value(value)
