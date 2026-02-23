from __future__ import annotations

from xdsl.dialects.builtin import (
    I32,
    IntegerAttr,
    StringAttr,
)
from xdsl.dialects.pdl import (
    OperationType,
    ValueType,
)
from xdsl.dialects.pdl_region import (
    RegionType,
)
from xdsl.ir import (
    Dialect,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    var_operand_def,
)


@irdl_op_definition
class GetRegionOp(IRDLOperation):
    name = "pdl_interp_region.get_region"
    input_op = operand_def(OperationType)
    index = prop_def(IntegerAttr[I32])
    value = result_def(RegionType)

    assembly_format = "$index `of` $input_op `:` type($value) attr-dict"

    def __init__(self, index: int | IntegerAttr[I32], input_op: SSAValue) -> None:
        if isinstance(index, int):
            index = IntegerAttr.from_int_and_width(index, 32)
        super().__init__(
            properties={"index": index},
            operands=[input_op],
            result_types=[RegionType],
        )


@irdl_op_definition
class InlineRegionOp(IRDLOperation):
    name = "pdl_interp_region.inline_region"
    input_op = operand_def(OperationType)
    repl_values = var_operand_def(RegionType)

    assembly_format = (
        "$input_op `with` ` ` `(` ($repl_values^ `:` type($repl_values))? `)` attr-dict"
    )

    def __init__(self, input_op: SSAValue, repl_values: SSAValue[RegionType]) -> None:
        super().__init__(operands=[input_op, repl_values])


@irdl_op_definition
class ValueOfYieldOp(IRDLOperation):
    name = "pdl_interp_region.value_of_yield"
    input_op = operand_def(RegionType)
    value = result_def(ValueType)

    assembly_format = "$input_op attr-dict"

    def __init__(self, input_op: SSAValue) -> None:
        super().__init__(
            operands=[input_op],
            result_types=[ValueType],
        )


@irdl_op_definition
class DebugPrintStatement(IRDLOperation):
    name = "pdl_interp_region.debug_print"
    message = prop_def(StringAttr)

    assembly_format = "`message` `(` $message `)` attr-dict"

    def __init__(self, message: str | StringAttr) -> None:
        if isinstance(message, str):
            message = StringAttr(message)
        super().__init__(properties={"message": message})


PDLInterpRegion = Dialect(
    "pdl_interp_region",
    [
        GetRegionOp,
        InlineRegionOp,
        DebugPrintStatement,
        ValueOfYieldOp,
    ],
)
