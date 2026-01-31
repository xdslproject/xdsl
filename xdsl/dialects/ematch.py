from __future__ import annotations

from xdsl.dialects.pdl import (
    OperationType,
    RangeType,
    ValueType,
)
from xdsl.ir import (
    Dialect,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class GetClassValsOp(IRDLOperation):
    """
    Take a value and return all values in its equivalence class.

    If the value is an equivalence.class result, return the operands of the class,
    otherwise return a range containing the value itself.
    """

    name = "ematch.get_class_vals"
    value = operand_def(ValueType)
    result = result_def(RangeType[ValueType])

    assembly_format = "$value attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(
            operands=[value],
            result_types=[RangeType(ValueType())],
        )


@irdl_op_definition
class GetClassRepresentativeOp(IRDLOperation):
    """
    Get one of the values in the equivalence class of v.
    """

    name = "ematch.get_class_representative"
    value = operand_def(ValueType)
    result = result_def(ValueType)

    assembly_format = "$value attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(
            operands=[value],
            result_types=[ValueType()],
        )


@irdl_op_definition
class GetClassResultOp(IRDLOperation):
    """
    Get the equivalence.class result corresponding to the equivalence class of v.
    """

    name = "ematch.get_class_result"
    value = operand_def(ValueType)
    result = result_def(ValueType)

    assembly_format = "$value attr-dict"

    def __init__(self, value: SSAValue) -> None:
        super().__init__(
            operands=[value],
            result_types=[ValueType()],
        )


@irdl_op_definition
class GetClassResultsOp(IRDLOperation):
    """
    Get the equivalence.class results corresponding to the equivalence classes
    of a range of values.
    """

    name = "ematch.get_class_results"
    values = operand_def(RangeType[ValueType])
    result = result_def(RangeType[ValueType])

    assembly_format = "$values attr-dict"

    def __init__(self, values: SSAValue) -> None:
        super().__init__(
            operands=[values],
            result_types=[RangeType(ValueType())],
        )


@irdl_op_definition
class UnionOp(IRDLOperation):
    """
    Merge two values, an operation and a value range, or two value ranges
    into equivalence class(es).

    Supported operand type combinations:
    - (value, value): merge two values
    - (operation, range<value>): merge operation results with values
    - (range<value>, range<value>): merge two value ranges
    """

    name = "ematch.union"
    lhs = operand_def(ValueType | OperationType | RangeType[ValueType])
    rhs = operand_def(ValueType | RangeType[ValueType])

    assembly_format = "$lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict"

    def __init__(self, lhs: SSAValue, rhs: SSAValue) -> None:
        super().__init__(operands=[lhs, rhs])


@irdl_op_definition
class DedupOp(IRDLOperation):
    """
    Check if the operation already exists in the hashcons.

    If so, remove the new one and return the existing one.
    """

    name = "ematch.dedup"
    input_op = operand_def(OperationType)
    result_op = result_def(OperationType)

    assembly_format = "$input_op attr-dict"

    def __init__(self, input_op: SSAValue) -> None:
        super().__init__(
            operands=[input_op],
            result_types=[OperationType()],
        )


EMatch = Dialect(
    "ematch",
    [
        GetClassValsOp,
        GetClassRepresentativeOp,
        GetClassResultOp,
        GetClassResultsOp,
        UnionOp,
        DedupOp,
    ],
)
