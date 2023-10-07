from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Generic, TypeVar, cast, overload

from xdsl.dialects.builtin import (
    AnyFloat,
    AnyIntegerAttr,
    ContainerOf,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
)
from xdsl.dialects.llvm import FastMathAttr as LLVMFastMathAttr
from xdsl.ir import Attribute, Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import (
    AnyOf,
    ConstraintVar,
    IRDLOperation,
    Operand,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, Pure
from xdsl.utils.deprecation import deprecated
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))

_FloatTypeT = TypeVar("_FloatTypeT", bound=AnyFloat)

CMPI_COMPARISON_OPERATIONS = [
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
]

CMPF_COMPARISON_OPERATIONS = [
    "false",
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
    "true",
]


class FastMathFlagsAttr(LLVMFastMathAttr):
    """
    arith.fastmath is a mirror of LLVMs fastmath flags.
    """

    name = "arith.fastmath"


@irdl_op_definition
class Constant(IRDLOperation):
    name = "arith.constant"
    result: OpResult = result_def(Attribute)
    value: Attribute = prop_def(Attribute)

    traits = frozenset((ConstantLike(),))

    @overload
    def __init__(
        self, value: AnyIntegerAttr | FloatAttr[AnyFloat], value_type: None = None
    ) -> None:
        ...

    @overload
    def __init__(self, value: Attribute, value_type: Attribute) -> None:
        ...

    def __init__(
        self,
        value: AnyIntegerAttr | FloatAttr[AnyFloat] | Attribute,
        value_type: Attribute | None = None,
    ):
        if value_type is None:
            value = cast(AnyIntegerAttr | FloatAttr[AnyFloat], value)
            value_type = value.type
        super().__init__(
            operands=[], result_types=[value_type], properties={"value": value}
        )

    @staticmethod
    @deprecated("Please use Constant(attr, value_type)")
    def from_attr(attr: Attribute, value_type: Attribute) -> Constant:
        return Constant.create(result_types=[value_type], properties={"value": attr})

    @staticmethod
    def from_int_and_width(
        value: int | IntAttr, value_type: int | IntegerType | IndexType
    ) -> Constant:
        if isinstance(value_type, int):
            value_type = IntegerType(value_type)
        return Constant.create(
            result_types=[value_type],
            properties={"value": IntegerAttr(value, value_type)},
        )

    @staticmethod
    @deprecated("Please use Constant(attr) or Constant(FloatAttr(value, value_type))")
    def from_float_and_width(
        value: float | FloatAttr[_FloatTypeT], value_type: _FloatTypeT
    ) -> Constant:
        if isinstance(value, FloatAttr):
            value_attr = value
        else:
            value_attr = FloatAttr(value, value_type)
        return Constant.create(
            result_types=[value_type], properties={"value": value_attr}
        )

    def print(self, printer: Printer):
        printer.print_op_attributes(self.attributes)

        printer.print(" ")
        printer.print_attribute(self.value)

    @classmethod
    def parse(cls: type[Constant], parser: Parser) -> Constant:
        attrs = parser.parse_optional_attr_dict()

        p0 = parser.pos
        value = parser.parse_attribute()

        if not isa(value, AnyIntegerAttr | FloatAttr[AnyFloat]):
            parser.raise_error("Invalid constant value", p0, parser.pos)

        c = Constant(value)
        c.attributes.update(attrs)
        return c


_T = TypeVar("_T", bound=Attribute)


class BinaryOperation(IRDLOperation, Generic[_T]):
    """A generic base class for arith's binary operation.

    They all have two operands and one result of a same type."""

    T = Annotated[Attribute, ConstraintVar("T"), _T]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(operands=[operand1, operand2], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        rhs = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        (lhs, rhs) = parser.resolve_operands([lhs, rhs], 2 * [result_type], parser.pos)
        return cls(lhs, rhs, result_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.result.type)

    def __hash__(self) -> int:
        return id(self)


SignlessIntegerBinaryOp = BinaryOperation[Annotated[Attribute, signlessIntegerLike]]


class BinaryOperationWithFastMath(Generic[_T], BinaryOperation[_T]):
    fastmath = opt_prop_def(FastMathFlagsAttr)


FloatingPointLikeBinaryOp = BinaryOperationWithFastMath[
    Annotated[Attribute, floatingPointLike]
]

IntegerBinaryOp = BinaryOperation[IntegerType]


@irdl_op_definition
class Addi(SignlessIntegerBinaryOp):
    name = "arith.addi"

    traits = frozenset([Pure()])


@irdl_op_definition
class Muli(SignlessIntegerBinaryOp):
    name = "arith.muli"


@irdl_op_definition
class Subi(SignlessIntegerBinaryOp):
    name = "arith.subi"


@irdl_op_definition
class DivUI(SignlessIntegerBinaryOp):
    """
    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.
    """

    name = "arith.divui"


@irdl_op_definition
class DivSI(SignlessIntegerBinaryOp):
    """
    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.
    """

    name = "arith.divsi"


@irdl_op_definition
class FloorDivSI(SignlessIntegerBinaryOp):
    """
    Signed floor integer division. Rounds towards negative infinity i.e. `5 / -2 = -3`.
    """

    name = "arith.floordivsi"


@irdl_op_definition
class CeilDivSI(SignlessIntegerBinaryOp):
    name = "arith.ceildivsi"


@irdl_op_definition
class CeilDivUI(SignlessIntegerBinaryOp):
    name = "arith.ceildivui"


@irdl_op_definition
class RemUI(SignlessIntegerBinaryOp):
    name = "arith.remui"


@irdl_op_definition
class RemSI(SignlessIntegerBinaryOp):
    name = "arith.remsi"


@irdl_op_definition
class MinUI(SignlessIntegerBinaryOp):
    name = "arith.minui"


@irdl_op_definition
class MaxUI(SignlessIntegerBinaryOp):
    name = "arith.maxui"


@irdl_op_definition
class MinSI(SignlessIntegerBinaryOp):
    name = "arith.minsi"


@irdl_op_definition
class MaxSI(SignlessIntegerBinaryOp):
    name = "arith.maxsi"


@irdl_op_definition
class AndI(SignlessIntegerBinaryOp):
    name = "arith.andi"


@irdl_op_definition
class OrI(SignlessIntegerBinaryOp):
    name = "arith.ori"


@irdl_op_definition
class XOrI(SignlessIntegerBinaryOp):
    name = "arith.xori"


@irdl_op_definition
class ShLI(SignlessIntegerBinaryOp):
    """
    The `shli` operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.
    """

    name = "arith.shli"


@irdl_op_definition
class ShRUI(SignlessIntegerBinaryOp):
    """
    The `shrui` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as unsigned. The high order bits are
    always filled with zeros.
    """

    name = "arith.shrui"


@irdl_op_definition
class ShRSI(SignlessIntegerBinaryOp):
    """
    The `shrsi` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as signed. The high order bits in the
    output are filled with copies of the most-significant bit of the shifted
    value (which means that the sign of the value is preserved).
    """

    name = "arith.shrsi"


@dataclass
class ComparisonOperation:
    """
    A generic comparison operation, operation definitions inherit this class.

    The first argument to these comparison operations is the type of comparison
    being performed, the following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    """

    @staticmethod
    def _get_comparison_predicate(
        mnemonic: str, comparison_operations: dict[str, int]
    ) -> int:
        if mnemonic in comparison_operations:
            return comparison_operations[mnemonic]
        else:
            raise VerifyException(f"Unknown comparison mnemonic: {mnemonic}")

    @staticmethod
    def _validate_operand_types(operand1: SSAValue, operand2: SSAValue):
        if operand1.type != operand2.type:
            raise TypeError(
                f"Comparison operands must have same type, but "
                f"provided {operand1.type} and {operand2.type}"
            )


@irdl_op_definition
class Cmpi(IRDLOperation, ComparisonOperation):
    """
    The cmpi operation is a generic comparison for integer-like types. Its two
    arguments can be integers, vectors or tensors thereof as long as their types
    match. The operation produces an i1 for the former case, a vector or a
    tensor of i1 with the same shape as inputs in the other cases.

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Example:

    // Custom form of scalar "signed less than" comparison.
    %x = arith.cmpi slt, %lhs, %rhs : i32

    // Generic form of the same operation.
    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

    // Custom form of vector equality comparison.
    %x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

    // Generic form of the same operation.
    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
        : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
    """

    name = "arith.cmpi"
    predicate: AnyIntegerAttr = prop_def(AnyIntegerAttr)
    lhs: Operand = operand_def(signlessIntegerLike)
    rhs: Operand = operand_def(signlessIntegerLike)
    result: OpResult = result_def(IntegerType(1))

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        arg: int | str,
    ):
        operand1 = SSAValue.get(operand1)
        operand2 = SSAValue.get(operand2)
        Cmpi._validate_operand_types(operand1, operand2)

        if isinstance(arg, str):
            cmpi_comparison_operations = {
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            }
            arg = Cmpi._get_comparison_predicate(arg, cmpi_comparison_operations)

        return super().__init__(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            properties={"predicate": IntegerAttr.from_int_and_width(arg, 64)},
        )

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_identifier()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], 2 * [input_type], parser.pos
        )

        return cls(operand1, operand2, arg)

    def print(self, printer: Printer):
        printer.print(" ")

        printer.print_string(CMPI_COMPARISON_OPERATIONS[self.predicate.value.data])
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.lhs.type)


@irdl_op_definition
class Cmpf(IRDLOperation, ComparisonOperation):
    """
    The cmpf operation compares its two operands according to the float
    comparison rules and the predicate specified by the respective attribute.
    The predicate defines the type of comparison: (un)orderedness, (in)equality
    and signed less/greater than (or equal to) as well as predicates that are
    always true or false.  The operands must have the same type, and this type
    must be a float type, or a vector or tensor thereof.  The result is an i1,
    or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi,
    the operands are always treated as signed. The u prefix indicates
    *unordered* comparison, not unsigned comparison, so "une" means unordered or
    not equal. For the sake of readability by humans, custom assembly form for
    the operation uses a string-typed attribute for the predicate.  The value of
    this attribute corresponds to lower-cased name of the predicate constant,
    e.g., "one" means "ordered not equal".  The string representation of the
    attribute is merely a syntactic sugar and is converted to an integer
    attribute by the parser.

    Example:

    %r1 = arith.cmpf oeq, %0, %1 : f32
    %r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
    %r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
    """

    name = "arith.cmpf"
    predicate: AnyIntegerAttr = prop_def(AnyIntegerAttr)
    lhs: Operand = operand_def(floatingPointLike)
    rhs: Operand = operand_def(floatingPointLike)
    result: OpResult = result_def(IntegerType(1))

    def __init__(
        self,
        operand1: SSAValue | Operation,
        operand2: SSAValue | Operation,
        arg: int | str,
    ):
        operand1 = SSAValue.get(operand1)
        operand2 = SSAValue.get(operand2)

        Cmpf._validate_operand_types(operand1, operand2)

        if isinstance(arg, str):
            cmpf_comparison_operations = {
                "false": 0,
                "oeq": 1,
                "ogt": 2,
                "oge": 3,
                "olt": 4,
                "ole": 5,
                "one": 6,
                "ord": 7,
                "ueq": 8,
                "ugt": 9,
                "uge": 10,
                "ult": 11,
                "ule": 12,
                "une": 13,
                "uno": 14,
                "true": 15,
            }
            arg = Cmpf._get_comparison_predicate(arg, cmpf_comparison_operations)

        return super().__init__(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            properties={"predicate": IntegerAttr.from_int_and_width(arg, 64)},
        )

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_identifier()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], 2 * [input_type], parser.pos
        )

        return cls(operand1, operand2, arg)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_string(CMPF_COMPARISON_OPERATIONS[self.predicate.value.data])
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.lhs.type)


@irdl_op_definition
class Select(IRDLOperation):
    """
    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.
    """

    name = "arith.select"
    cond: Operand = operand_def(IntegerType(1))  # should be unsigned
    lhs: Operand = operand_def(Attribute)
    rhs: Operand = operand_def(Attribute)
    result: OpResult = result_def(Attribute)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.cond.type != IntegerType(1):
            raise VerifyException("Condition has to be of type !i1")
        if self.lhs.type != self.rhs.type or self.rhs.type != self.result.type:
            raise VerifyException("expect all input and output types to be equal")

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        operand3: Operation | SSAValue,
    ):
        operand2 = SSAValue.get(operand2)
        return super().__init__(
            operands=[operand1, operand2, operand3], result_types=[operand2.type]
        )

    @classmethod
    def parse(cls, parser: Parser):
        cond = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        (cond, operand1, operand2) = parser.resolve_operands(
            [cond, operand1, operand2],
            [IntegerType(1), result_type, result_type],
            parser.pos,
        )

        return cls(cond, operand1, operand2)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.cond)
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class Addf(FloatingPointLikeBinaryOp):
    name = "arith.addf"


@irdl_op_definition
class Subf(FloatingPointLikeBinaryOp):
    name = "arith.subf"


@irdl_op_definition
class Mulf(FloatingPointLikeBinaryOp):
    name = "arith.mulf"


@irdl_op_definition
class Divf(FloatingPointLikeBinaryOp):
    name = "arith.divf"


@irdl_op_definition
class Negf(IRDLOperation):
    name = "arith.negf"
    fastmath: FastMathFlagsAttr | None = opt_prop_def(FastMathFlagsAttr)
    operand: Operand = operand_def(floatingPointLike)
    result: OpResult = result_def(floatingPointLike)

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.type],
        )

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()
        input = parser.resolve_operand(input, result_type)
        return cls(input)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.operand)
        printer.print(" : ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class Maxf(FloatingPointLikeBinaryOp):
    name = "arith.maxf"


@irdl_op_definition
class Minf(FloatingPointLikeBinaryOp):
    name = "arith.minf"


@irdl_op_definition
class IndexCastOp(IRDLOperation):
    name = "arith.index_cast"

    input: Operand = operand_def()

    result: OpResult = result_def()

    def __init__(self, input_arg: SSAValue | Operation, target_type: Attribute):
        return super().__init__(operands=[input_arg], result_types=[target_type])


@irdl_op_definition
class FPToSIOp(IRDLOperation):
    name = "arith.fptosi"

    input: Operand = operand_def(AnyFloat)
    result: OpResult = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])


@irdl_op_definition
class SIToFPOp(IRDLOperation):
    name = "arith.sitofp"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(AnyFloat)

    def __init__(self, op: SSAValue | Operation, target_type: AnyFloat):
        return super().__init__(operands=[op], result_types=[target_type])


@irdl_op_definition
class ExtFOp(IRDLOperation):
    name = "arith.extf"

    input: Operand = operand_def(AnyFloat)
    result: OpResult = result_def(AnyFloat)

    def __init__(self, op: SSAValue | Operation, target_type: AnyFloat):
        return super().__init__(operands=[op], result_types=[target_type])

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_keyword("to")
        result_type = parser.parse_type()
        [input] = parser.resolve_operands([input], [input_type], parser.pos)
        result_float_type = cast(AnyFloat, result_type)
        return cls(input, result_float_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.input)
        printer.print(" : ")
        printer.print_attribute(self.input.type)
        printer.print(" to ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class TruncFOp(IRDLOperation):
    name = "arith.truncf"

    input: Operand = operand_def(AnyFloat)
    result: OpResult = result_def(AnyFloat)

    def __init__(self, op: SSAValue | Operation, target_type: AnyFloat):
        return super().__init__(operands=[op], result_types=[target_type])

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_keyword("to")
        result_type = parser.parse_type()
        [input] = parser.resolve_operands([input], [input_type], parser.pos)
        result_float_type = cast(AnyFloat, result_type)
        return cls(input, result_float_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.input)
        printer.print(" : ")
        printer.print_attribute(self.input.type)
        printer.print(" to ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class TruncIOp(IRDLOperation):
    name = "arith.trunci"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])

    def verify_(self) -> None:
        assert isinstance(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if not self.result.type.width.data < self.input.type.width.data:
            raise VerifyException(
                "Destination bit-width must be smaller than the input bit-width"
            )

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_keyword("to")
        result_type = parser.parse_type()
        [input] = parser.resolve_operands([input], [input_type], parser.pos)
        result_int_type = cast(IntegerType, result_type)
        return cls(input, result_int_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.input)
        printer.print(" : ")
        printer.print_attribute(self.input.type)
        printer.print(" to ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class ExtSIOp(IRDLOperation):
    name = "arith.extsi"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])

    def verify_(self) -> None:
        assert isinstance(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if not self.result.type.width.data > self.input.type.width.data:
            raise VerifyException(
                "Destination bit-width must be larger than the input bit-width"
            )

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_keyword("to")
        result_type = parser.parse_type()
        [input] = parser.resolve_operands([input], [input_type], parser.pos)
        result_int_type = cast(IntegerType, result_type)
        return cls(input, result_int_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.input)
        printer.print(" : ")
        printer.print_attribute(self.input.type)
        printer.print(" to ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class ExtUIOp(IRDLOperation):
    name = "arith.extui"

    input: Operand = operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        return super().__init__(operands=[op], result_types=[target_type])

    def verify_(self) -> None:
        assert isinstance(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if not self.result.type.width.data > self.input.type.width.data:
            raise VerifyException(
                "Destination bit-width must be larger than the input bit-width"
            )

    @classmethod
    def parse(cls, parser: Parser):
        input = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        parser.parse_keyword("to")
        result_type = parser.parse_type()
        [input] = parser.resolve_operands([input], [input_type], parser.pos)
        result_int_type = cast(IntegerType, result_type)
        return cls(input, result_int_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.input)
        printer.print(" : ")
        printer.print_attribute(self.input.type)
        printer.print(" to ")
        printer.print_attribute(self.result.type)


Arith = Dialect(
    [
        Constant,
        # Integer-like
        Addi,
        Subi,
        Muli,
        DivUI,
        DivSI,
        FloorDivSI,
        CeilDivSI,
        CeilDivUI,
        RemUI,
        RemSI,
        MinSI,
        MaxSI,
        MinUI,
        MaxUI,
        # Float-like
        Addf,
        Subf,
        Mulf,
        Divf,
        Negf,
        # Comparison/Condition
        Cmpi,
        Cmpf,
        Select,
        # Logical
        AndI,
        OrI,
        XOrI,
        # Shift
        ShLI,
        ShRUI,
        ShRSI,
        # Min/Max
        Minf,
        Maxf,
        # Casts
        IndexCastOp,
        FPToSIOp,
        SIToFPOp,
        ExtFOp,
        TruncFOp,
        TruncIOp,
        ExtSIOp,
        ExtUIOp,
    ],
    [
        FastMathFlagsAttr,
    ],
)
