from __future__ import annotations
from dataclasses import dataclass, field
from io import StringIO

from xdsl.ir import IRNode, Block, Operation, Region
from xdsl.utils.exceptions import DiagnosticException


@dataclass
class Diagnostic:
    op_messages: dict[Operation, list[str]] = field(default_factory=dict)

    def add_message(self, op: Operation, message: str) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(message)

    def raise_exception(
        self,
        message: str,
        ir: IRNode,
        exception_type: type[Exception] = DiagnosticException,
    ) -> None:
        """Raise an exception, that will also print all messages in the IR."""
        from xdsl.printer import Printer

        f = StringIO()
        p = Printer(stream=f, diagnostic=self)
        toplevel = ir.get_toplevel_object()
        if isinstance(toplevel, Operation):
            p.print_op(toplevel)
            print("\n", file=f)
        elif isinstance(toplevel, Block):
            p.print_block(toplevel)
        elif isinstance(toplevel, Region):
            # TOFIX: Is that ever used. Revisit the whole exception
            p._print_region(toplevel)  # TOFIX #type: ignore
        else:
            assert "xDSL internal error: get_toplevel_object returned unknown construct"

        raise exception_type(message + "\n\n" + f.getvalue())
