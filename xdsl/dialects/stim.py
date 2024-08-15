from xdsl.dialects.builtin import (
    AnyMemRefType,
    BoolAttr,
    StringAttr,
    IntegerAttr,
    MemRefType,
)

from xdsl.irdl import (
    IRDLOperation,
    ParameterDef,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
)

@irdl_op_definition
class StimModuleOp(IRDLOperation):
    """
    Base operation containing a stim program
    """

    name = "stim.circuit"

    body = region_def()

    assembly_format = "$body attr-dict"

    def __init__(self, body: body):
        super().__init__(body=[body])

Stim = Dialect(
    "stim",
    #first list operations to include in the dialect
    [
        StimModuleOp,

    ]
)