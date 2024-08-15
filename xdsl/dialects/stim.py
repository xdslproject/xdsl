
from xdsl.ir.core import Dialect, Region

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
)
from xdsl.irdl.operations import region_def

@irdl_op_definition
class StimModuleOp(IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def()

    assembly_format = "$body attr-dict"

    def __init__(self, body: Region):
        super().__init__(regions =[body])

Stim = Dialect(
    "stim",
    #first list operations to include in the dialect
    [
        StimModuleOp,

    ]
)