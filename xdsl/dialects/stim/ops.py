from abc import ABC
from io import StringIO

from xdsl.ir import Region
from xdsl.irdl import (
    IRDLOperation,
    PyRDLOpDefinitionError,
    irdl_op_definition,
    region_def,
)

from .stim_printer_parser import StimPrintable, StimPrinter


class StimOp(IRDLOperation, ABC):
    def print_stim(self, printer: StimPrinter) -> None:
        raise (PyRDLOpDefinitionError("print_stim not implemented!"))


@irdl_op_definition
class StimCircuitOp(StimOp, IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def("single_block")

    assembly_format = "attr-dict-with-keyword $body"

    def __init__(self, body: Region):
        super().__init__(regions=[body])

    def verify(self, verify_nested_ops: bool = True) -> None:
        return

    def print_stim(self, printer: StimPrinter):
        for op in self.body.block.ops:
            if not isinstance(op, StimPrintable):
                raise ValueError(f"Cannot print in stim format: {op}")
            op.print_stim(printer)
        printer.print_string("")

    def stim(self) -> str:
        io = StringIO()
        printer = StimPrinter(io)
        self.print_stim(printer)
        res = io.getvalue()
        return res
