from xdsl.ir import Attribute, TypeAttribute
from xdsl.printer import Printer
from xdsl.utils.color_printer import ColorPrinter
from xdsl.utils.colors import Colors


class SyntaxPrinter(Printer, ColorPrinter):
    """
    A printer for printing syntax-highlighted mlir code to a terminal.
    """

    def print_op_name(self, name: str):
        with self.colored(Colors.CYAN):
            return super().print_op_name(name)

    def print_attribute(self, attribute: Attribute):
        if isinstance(attribute, TypeAttribute):
            with self.colored(Colors.GREEN):
                return super().print_attribute(attribute)
        return super().print_attribute(attribute)
