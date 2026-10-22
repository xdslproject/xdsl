from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import ClassVar, NoReturn

from xdsl.ir import Block, IRNode, Operation, Region


@dataclass
class Diagnostic:
    colored: ClassVar = False
    op_messages: dict[Operation, list[str]] = field(
        default_factory=dict[Operation, list[str]]
    )

    def add_message(self, op: Operation, message: str) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(message)

    def raise_exception(self, ir: IRNode, underlying_error: Exception) -> NoReturn:
        """Raise an exception, that will also print all messages in the IR."""
        if self.colored:
            from xdsl.syntax_printer import SyntaxPrinter as DiagnosticPrinter
        else:
            from xdsl.printer import Printer as DiagnosticPrinter

        f = StringIO()
        p = DiagnosticPrinter(stream=f, diagnostic=self, print_generic_format=True)
        toplevel = ir.get_toplevel_object()
        match toplevel:
            case Operation():
                p.print_op(toplevel)
                p.print_string("\n")
            case Block():
                p.print_block(toplevel)
                p.print_string("\n")
            case Region():
                p.print_region(toplevel)

        underlying_error.add_note(f.getvalue())

        raise underlying_error
