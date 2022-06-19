from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Type, Union

from xdsl.ir import Block, Operation, Region


class DiagnosticException(Exception):
    ...


@dataclass
class Diagnostic:
    op_messages: Dict[Operation, List[str]] = field(default_factory=dict)

    def add_message(self, op: Operation, message: str) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(message)

    def raise_exception(
            self,
            message: str,
            ir: Union[Operation, Block, Region],
            exception_type: Type[Exception] = DiagnosticException) -> None:
        """Raise an exception, that will also print all messages in the IR."""
        from xdsl.printer import Printer
        f = StringIO()
        p = Printer(stream=f, diagnostic=self)
        toplevel = ir.get_toplevel_object()
        if isinstance(toplevel, Operation):
            p.print_op(toplevel)
        elif isinstance(toplevel, Block):
            p._print_named_block(toplevel)  # type: ignore
        elif isinstance(toplevel, Region):
            p.print_region(toplevel)
        else:
            assert "xDSL internal error: get_toplevel_object returned unknown construct"
        raise exception_type(message + "\n\n" + f.getvalue())
