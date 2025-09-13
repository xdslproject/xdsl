"""
This dialect provides operations for marking regions of interest for cost modeling.

The primary use case is to delineate sections of code for performance
analysis, without altering the program's semantics (no-ops).
"""

from xdsl.backend.assembly_printer import OneLineAssemblyPrintable
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
)


@irdl_op_definition
class BeginMCARegionOfInterestOp(IRDLOperation, OneLineAssemblyPrintable):
    """
    Marks the beginning of a region of interest for `llvm-mca`
    """

    name = "cost_model.begin_mca_region_of_interest"

    assembly_format = "attr-dict"

    def assembly_line(self) -> str:
        return "# LLVM-MCA-BEGIN"


@irdl_op_definition
class StopMCARegionOfInterestOp(IRDLOperation, OneLineAssemblyPrintable):
    """
    Marks the end of a region of interest for `llvm-mca`.
    """

    name = "cost_model.stop_mca_region_of_interest"

    assembly_format = "attr-dict"

    def assembly_line(self) -> str:
        return "# LLVM-MCA-END"


CostModel = Dialect(
    "cost_model",
    [
        BeginMCARegionOfInterestOp,
        StopMCARegionOfInterestOp,
    ],
)
