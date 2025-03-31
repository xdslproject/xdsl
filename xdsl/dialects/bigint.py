"""Dialect for arbitrary-precision integers."""

import abc
from typing import ClassVar

from xdsl.dialects.builtin import ContainerOf
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
    VarConstraint,
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


bigIntegerLike = ContainerOf(BigIntegerType)


class BigIntegerBinaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", bigIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

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
class AddBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.add"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs + rhs


@irdl_op_definition
class SubBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.sub"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs - rhs


@irdl_op_definition
class MulBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.mul"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs * rhs


@irdl_op_definition
class FloorDivBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.floordiv"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs // rhs


@irdl_op_definition
class ModBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.mod"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs % rhs


@irdl_op_definition
class PowBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.mod"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs % rhs


@irdl_op_definition
class LShiftBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.lshift"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs << rhs


@irdl_op_definition
class RShiftBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.rshift"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs >> rhs


@irdl_op_definition
class BitOrBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.bitor"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs | rhs


@irdl_op_definition
class BitXorBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.bitxor"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs ^ rhs


@irdl_op_definition
class BitAndBigIntOp(BigIntegerBinaryOperation):
    name = "bigint.bitand"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )

    @staticmethod
    def py_operation(lhs: int, rhs: int) -> int:
        return lhs & rhs


# @irdl_op_definition
# class DivBigIntOp(IRDLOperation):
#     name = "bigint.div"

#     T: ClassVar = VarConstraint("T", bigIntegerLike)
#     R: ClassVar = VarConstraint("R", floatingPointLike)

#     lhs = operand_def(T)
#     rhs = operand_def(T)
#     result = result_def(R)

#     assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

#     traits = traits_def(
#         Pure(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> float:
#         """Performs a python function corresponding to this operation."""
#         return lhs / rhs

#     def __init__(
#         self,
#         operand1: Operation | SSAValue,
#         operand2: Operation | SSAValue,
#         result_type: Attribute | None = None,
#     ):
#         if result_type is None:
#             result_type = SSAValue.get(operand1).type
#         super().__init__(operands=[operand1, operand2], result_types=[result_type])


# class BigIntegerComparisonOperation(IRDLOperation, abc.ABC):
#     T: ClassVar = VarConstraint("T", bigIntegerLike)
#     R: ClassVar = VarConstraint("R", boolLike)

#     lhs = operand_def(T)
#     rhs = operand_def(T)
#     result = result_def(R)

#     assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         """Performs a python function corresponding to this operation."""
#         ...

#     def __init__(
#         self,
#         operand1: Operation | SSAValue,
#         operand2: Operation | SSAValue,
#         result_type: Attribute | None = None,
#     ):
#         if result_type is None:
#             result_type = SSAValue.get(operand1).type
#         super().__init__(operands=[operand1, operand2], result_types=[result_type])


# @irdl_op_definition
# class EqBigIntOp(BigIntegerComparisonOperation):
#     name = "bigint.eq"

#     traits = traits_def(
#         Pure(),
#         Commutative(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         return lhs == rhs


# @irdl_op_definition
# class NeqBigIntOp(BigIntegerComparisonOperation):
#     name = "bigint.neq"

#     traits = traits_def(
#         Pure(),
#         Commutative(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         return lhs != rhs


# @irdl_op_definition
# class GtBigIntOp(BigIntegerComparisonOperation):
#     name = "bigint.gt"

#     traits = traits_def(
#         Pure(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         return lhs > rhs


# @irdl_op_definition
# class GteBigIntOp(BigIntegerComparisonOperation):
#     name = "bigint.gte"

#     traits = traits_def(
#         Pure(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         return lhs >= rhs


# @irdl_op_definition
# class LtBigIntOp(BigIntegerComparisonOperation):
#     name = "bigint.lt"

#     traits = traits_def(
#         Pure(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         return lhs < rhs


# @irdl_op_definition
# class LteBigIntOp(BigIntegerComparisonOperation):
#     name = "bigint.lte"

#     traits = traits_def(
#         Pure(),
#     )

#     @staticmethod
#     def py_operation(lhs: int, rhs: int) -> int:
#         return lhs <= rhs


bigint = BigIntegerType()

BigInt = Dialect(
    "bigint",
    [
        AddBigIntOp,
        SubBigIntOp,
        MulBigIntOp,
        FloorDivBigIntOp,
        ModBigIntOp,
        PowBigIntOp,
        LShiftBigIntOp,
        RShiftBigIntOp,
        BitOrBigIntOp,
        BitXorBigIntOp,
        BitAndBigIntOp,
        # DivBigIntOp,
        # EqBigIntOp,
        # NeqBigIntOp,
        # GtBigIntOp,
        # GteBigIntOp,
        # LtBigIntOp,
        # LteBigIntOp,
    ],
    [
        BigIntegerType,
    ],
)
