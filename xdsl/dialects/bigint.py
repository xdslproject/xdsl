"""Dialect for unlimited precision integers with Python `int` semantics."""

import abc
from collections.abc import Sequence

from xdsl.dialects.builtin import IntAttr, f64, i1
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
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    Commutative,
    ConstantLike,
    HasFolder,
    Pure,
    SameOperandsAndResultType,
)


@irdl_attr_definition
class BigIntegerType(ParametrizedAttribute, TypeAttribute):
    """Type for unlimited precision integers, with Python `int` semantics."""

    name = "bigint.bigint"


bigint = BigIntegerType()


class BinaryOperation(IRDLOperation, abc.ABC):
    """Binary operation where all operands and results are `bigint`s."""

    lhs = operand_def(bigint)
    rhs = operand_def(bigint)
    result = result_def(bigint)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
    ):
        super().__init__(operands=[operand1, operand2], result_types=[bigint])


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "bigint.constant"
    result = result_def(bigint)
    value = prop_def(IntAttr)

    traits = traits_def(ConstantLike(), HasFolder(), Pure())

    @classmethod
    def parse(cls, parser: Parser):
        integer = parser.parse_integer()
        attrDict = parser.parse_optional_attr_dict()
        return cls(value=IntAttr(integer), attrs=attrDict)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_int(self.value.data)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)

    def __init__(
        self,
        value: IntAttr | int,
        attrs: dict[str, Attribute] | None = None,
    ):
        if isinstance(value, int):
            value = IntAttr(value)

        super().__init__(
            operands=[],
            result_types=[bigint],
            properties={"value": value},
            attributes=attrs,
        )

    def fold(self) -> Sequence[SSAValue | Attribute] | None:
        return (self.value,)


@irdl_op_definition
class AddOp(BinaryOperation):
    """Add two `bigint`s."""

    name = "bigint.add"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class SubOp(BinaryOperation):
    """Subtract two `bigint`s."""

    name = "bigint.sub"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class MulOp(BinaryOperation):
    """Multiply two `bigint`s."""

    name = "bigint.mul"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class FloorDivOp(BinaryOperation):
    """Floor divide two `bigint`s, rounding down to the nearest integer."""

    name = "bigint.floordiv"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class ModOp(BinaryOperation):
    """Modulo two `bigint`s, taking the sign of the divisor."""

    name = "bigint.mod"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class PowOp(BinaryOperation):
    """Exponentiate a `bigint` by another."""

    name = "bigint.pow"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class LShiftOp(BinaryOperation):
    """Left shift a `bigint` by another."""

    name = "bigint.lshift"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class RShiftOp(BinaryOperation):
    """Right shift a `bigint` by another."""

    name = "bigint.rshift"

    traits = traits_def(
        Pure(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class BitOrOp(BinaryOperation):
    """Bitwise OR a `bigint` with another."""

    name = "bigint.bitor"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class BitXorOp(BinaryOperation):
    """Bitwise XOR a `bigint` with another."""

    name = "bigint.bitxor"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class BitAndOp(BinaryOperation):
    """Bitwise AND a `bigint` with another."""

    name = "bigint.bitand"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


@irdl_op_definition
class DivOp(IRDLOperation):
    """Divide two `bigint`s, yielding a 64-bit floating point type.

    Note that this operation follows Python semantics, for example by rounding
    to minus infinity.
    """

    name = "bigint.div"

    lhs = operand_def(bigint)
    rhs = operand_def(bigint)
    result = result_def(f64)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    traits = traits_def(
        Pure(),
    )

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
    ):
        super().__init__(operands=[operand1, operand2], result_types=[f64])


class ComparisonOperation(IRDLOperation, abc.ABC):
    """Binary operation comparing two `bigint`s and returning a boolean."""

    lhs = operand_def(bigint)
    rhs = operand_def(bigint)
    result = result_def(i1)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
    ):
        super().__init__(operands=[operand1, operand2], result_types=[i1])


@irdl_op_definition
class EqOp(ComparisonOperation):
    """Check equality of two `bigint`s."""

    name = "bigint.eq"

    traits = traits_def(
        Pure(),
        Commutative(),
    )


@irdl_op_definition
class NeqOp(ComparisonOperation):
    """Check inequality of two `bigint`s."""

    name = "bigint.neq"

    traits = traits_def(
        Pure(),
        Commutative(),
    )


@irdl_op_definition
class GtOp(ComparisonOperation):
    """Check if one `bigint` is greater than another."""

    name = "bigint.gt"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class GteOp(ComparisonOperation):
    """Check if one `bigint` is greater than or equal to another."""

    name = "bigint.gte"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class LtOp(ComparisonOperation):
    """Check if one `bigint` is less than another."""

    name = "bigint.lt"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class LteOp(ComparisonOperation):
    """Check if one `bigint` is less than or equal to another."""

    name = "bigint.lte"

    traits = traits_def(
        Pure(),
    )


BigInt = Dialect(
    "bigint",
    [
        ConstantOp,
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
