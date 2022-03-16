from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field

from xdsl.ir import Block, Operation, Region
from io import StringIO


@dataclass
class Diagnostic:
    op_messages: Dict[Operation, List[str]] = field(default_factory=dict)

    def add_message(self, op: Operation, message: str) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(message)

    def raise_exception(self,
                        message,
                        ir: Union[Operation, Block, Region],
                        exception_type=Exception) -> None:
        """Raise an exception, that will also print all messages in the IR."""
        from xdsl.printer import Printer
        f = StringIO()
        p = Printer(stream=f, diagnostic=self)
        toplevel = ir.get_toplevel_object()
        if isinstance(toplevel, Operation):
            p.print_op(toplevel)
        elif isinstance(toplevel, Block):
            p._print_named_block(toplevel)
        elif isinstance(toplevel, Region):
            p._print_region(toplevel)
        else:
            assert "xDSL internal error: get_toplevel_object returned unknown construct"

        raise exception_type(message + "\n\n" + f.getvalue())
