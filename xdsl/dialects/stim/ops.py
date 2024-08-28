from abc import ABC
from io import StringIO
from xdsl.dialects.stim.stim_printer_parser import StimPrinter
from xdsl.ir import Region
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    PyRDLOpDefinitionError,
    region_def,
)

class StimOp(IRDLOperation, ABC):
    def print_stim(self, printer: StimPrinter) -> None:
        raise(PyRDLOpDefinitionError("print_stim not implemented!"))

@irdl_op_definition
class StimCircuitOp(StimOp, IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def('single_block')

    assembly_format = "attr-dict-with-keyword $body"

    def __init__(self, body: Region):
        super().__init__(regions=[body])

    def verify(self, verify_nested_ops: bool = True) -> None:
        for op in self.body.ops:
            if not isinstance(op, StimOp):
                raise (PyRDLOpDefinitionError(f"All operations in a stim circuit must sublcass StimOp, found {op.name}"))
       
    def print_stim(self, printer: StimPrinter):
        for op in self.body.block.ops:
            op.print
            
    def stim(self) -> str:
        io = StringIO()
        printer = StimPrinter(io)
        self.print_stim(printer)
        res = io.getvalue()
        return res