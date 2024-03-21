from __future__ import annotations

import abc
from dataclasses import dataclass, field
from io import StringIO
from typing import NoReturn, TYPE_CHECKING

from xdsl.ir import Block, IRNode, Operation, Region
from xdsl.utils.exceptions import DiagnosticException

if TYPE_CHECKING:
    from xdsl.printer import Printer


class OperationInformation(abc.ABC):
    @abc.abstractmethod
    def get_info(self, printer: Printer) -> str:
        pass


class StringInformation(OperationInformation):
    def __init__(self, msg: str):
        self.msg = msg
    def get_info(self, printer: Printer) -> str:
        return self.msg


@dataclass
class Diagnostic:
    op_messages: dict[Operation, list] = field(default_factory=dict)

    def add_message(self, op: Operation, message: str) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(StringInformation(message))

    def add_func(self, op: Operation, op_information: OperationInformation) -> None:
        """Add a message to an operation."""
        self.op_messages.setdefault(op, []).append(op_information)

    def raise_exception(
        self,
        message: str,
        ir: IRNode,
        exception_type: type[Exception] = DiagnosticException,
        underlying_error: Exception | None = None,
    ) -> NoReturn:
        """Raise an exception, that will also print all messages in the IR."""
        from xdsl.printer import Printer

        f = StringIO()
        p = Printer(stream=f, diagnostic=self, print_generic_format=True)
        toplevel = ir.get_toplevel_object()
        if isinstance(toplevel, Operation):
            p.print_op(toplevel)
            print("\n", file=f)
        elif isinstance(toplevel, Block):
            p.print_block(toplevel)
        elif isinstance(toplevel, Region):
            p.print_region(toplevel)
        else:
            assert "xDSL internal error: get_toplevel_object returned unknown construct"

        raise exception_type(message + "\n\n" + f.getvalue()) from underlying_error
