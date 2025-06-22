from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import NoReturn

from xdsl.ir import Block, IRNode, Operation, Region


@dataclass
class Diagnostic:
    op_messages: dict[Operation, list[str]] = field(
        default_factory=dict[Operation, list[str]]
    )

    def add_message(self, op: Operation, message: str) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(message)

    def raise_exception(self, ir: IRNode, underlying_error: Exception) -> NoReturn:
        """Raise an exception, that will also print all messages in the IR."""
        from xdsl.printer import Printer

        f = StringIO()
        p = Printer(stream=f, diagnostic=self, print_generic_format=True)
        toplevel = ir.get_toplevel_object()
        if isinstance(toplevel, Operation):
            p.print_op(toplevel)
            p.print_string("\n")
        elif isinstance(toplevel, Block):
            p.print_block(toplevel)
        elif isinstance(toplevel, Region):
            p.print_region(toplevel)
        else:
            assert "xDSL internal error: get_toplevel_object returned unknown construct"

        # __notes__ only in 3.11 and above
        if hasattr(underlying_error, "add_note"):
            # Use official API if present
            underlying_error.add_note(f.getvalue())
        else:
            # Add our own __notes__ if not
            if not hasattr(underlying_error, "__notes__"):
                underlying_error.__notes__ = []
            underlying_error.__notes__.append(f.getvalue())

        raise underlying_error
