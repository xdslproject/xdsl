from xdsl.dialects import builtin
from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_attr_definition
class ListType(ParametrizedAttribute, TypeAttribute):
    name = "list.list"
    elem_type: builtin.IntegerType


@irdl_op_definition
class LengthOp(IRDLOperation):
    name = "list.length"

    li = operand_def(ListType)
    result = result_def(builtin.i32)

    def __init__(self, li: SSAValue):
        super().__init__(
            operands=[li],
            result_types=[builtin.i32],
        )

    assembly_format = "$li attr-dict `:` type($li) `->` type($result)"


@irdl_op_definition
class RangeOp(IRDLOperation):
    name = "list.range"

    lower = operand_def(builtin.i32)
    upper = operand_def(builtin.i32)
    result = result_def(ListType)

    def __init__(self, lower: SSAValue, upper: SSAValue, result_type: ListType):
        super().__init__(
            operands=[lower, upper],
            result_types=[result_type],
        )

    assembly_format = "$lower `to` $upper attr-dict `:` type($result)"


LIST_DIALECT = Dialect("list", [LengthOp, RangeOp], [ListType])
