from xdsl.ir import Operation, SSAValue
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

    def print_op(self, op: Operation) -> None:
        with self.colored(Colors.RED if op in self.diagnostic.op_messages else None):
            return super().print_op(op)
