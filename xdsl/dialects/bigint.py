"""Dialect for arbitrary-precision integers."""

import abc

from xdsl.dialects.builtin import f64, i1
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
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
    traits_def,
)
from xdsl.traits import Commutative, Pure, SameOperandsAndResultType


@irdl_attr_definition
class BigIntegerType(ParametrizedAttribute, TypeAttribute):
    """
    Type for arbitrary-precision integers (bigints), such as those in Python.
    """

    name = "bigint.bigint"


bigint = BigIntegerType()


class BinaryOperation(IRDLOperation, abc.ABC):
    lhs = operand_def(bigint)
    rhs = operand_def(bigint)
    result = result_def(bigint)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        """Performs a python function corresponding to this operation."""
        ...

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(operands=[operand1, operand2], result_types=[result_type])


@irdl_op_definition
class AddOp(BinaryOperation):
    name = "bigint.add"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class SubOp(BinaryOperation):
    name = "bigint.sub"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class MulOp(BinaryOperation):
    name = "bigint.mul"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class FloorDivOp(BinaryOperation):
    name = "bigint.floordiv"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class ModOp(BinaryOperation):
    name = "bigint.mod"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class PowOp(BinaryOperation):
    name = "bigint.pow"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class LShiftOp(BinaryOperation):
    name = "bigint.lshift"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class RShiftOp(BinaryOperation):
    name = "bigint.rshift"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class BitOrOp(BinaryOperation):
    name = "bigint.bitor"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class BitXorOp(BinaryOperation):
    name = "bigint.bitxor"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class BitAndOp(BinaryOperation):
    name = "bigint.bitand"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class DivOp(IRDLOperation):
    name = "bigint.div"

    lhs = operand_def(bigint)
    rhs = operand_def(bigint)
    result = result_def(f64)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    traits = traits_def(
        Pure(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> float:
        """Performs a python function corresponding to this operation."""
        return lhs / rhs

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(operands=[operand1, operand2], result_types=[result_type])


class BigIntegerComparisonOperation(IRDLOperation, abc.ABC):
    lhs = operand_def(bigint)
    rhs = operand_def(bigint)
    result = result_def(i1)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        """Performs a python function corresponding to this operation."""
        ...

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(operands=[operand1, operand2], result_types=[result_type])


@irdl_op_definition
class EqOp(BigIntegerComparisonOperation):
    name = "bigint.eq"

    traits = traits_def(
        Pure(),
        Commutative(),
    )


@irdl_op_definition
class NeqOp(BigIntegerComparisonOperation):
    name = "bigint.neq"

    traits = traits_def(
        Pure(),
        Commutative(),
    )


@irdl_op_definition
class GtOp(BigIntegerComparisonOperation):
    name = "bigint.gt"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class GteOp(BigIntegerComparisonOperation):
    name = "bigint.gte"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class LtOp(BigIntegerComparisonOperation):
    name = "bigint.lt"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class LteOp(BigIntegerComparisonOperation):
    name = "bigint.lte"

    traits = traits_def(
        Pure(),
    )


BigInt = Dialect(
    "bigint",
    [
        AddOp,
        SubOp,
        MulOp,
        FloorDivOp,
        ModOp,
        PowOp,
        LShiftOp,
        RShiftOp,
        BitOrOp,
        BitXorOp,
        BitAndOp,
        DivOp,
        EqOp,
        NeqOp,
        GtOp,
        GteOp,
        LtOp,
        LteOp,
    ],
    [
        BigIntegerType,
    ],
)
