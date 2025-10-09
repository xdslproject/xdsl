from xdsl.ir import Attribute, Block, Operation, Region, TypeAttribute
from xdsl.printer import Printer
from xdsl.utils.color_printer import ColorPrinter
from xdsl.utils.colors import Colors


class SyntaxPrinter(Printer, ColorPrinter):
    """
    A printer for printing syntax-highlighted mlir code to a terminal.
    """

    def print_op_name(self, name: str):
        dialect = name.split(".")[0]
        if dialect in ["scf", "cf", "func"]:
            color = Colors.MAGENTA
        else:
            color = Colors.CYAN
        with self.colored(color):
            return super().print_op_name(name)

    def print_attribute(self, attribute: Attribute):
        if isinstance(attribute, TypeAttribute):
            with self.colored(Colors.GREEN):
                return super().print_attribute(attribute)
        return super().print_attribute(attribute)


def pprint(obj: Operation | Region | Block):
    printer = SyntaxPrinter()

    if isinstance(obj, Operation):
        printer.print_op(obj)
    elif isinstance(obj, Region):
        printer.print_region(obj)
    elif isinstance(obj, Block):
        printer.print_block(obj)
