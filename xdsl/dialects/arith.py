from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Generic, TypeVar, Set, Optional

from xdsl.dialects.builtin import (
    ContainerOf,
    Float16Type,
    Float64Type,
    IndexType,
    IntAttr,
    IntegerType,
    Float32Type,
    IntegerAttr,
    FloatAttr,
    Attribute,
    AnyFloat,
    AnyIntegerAttr,
)
from xdsl.ir import Operation, SSAValue, Dialect, OpResult, Data
from xdsl.irdl import (
    AnyOf,
    ConstraintVar,
    irdl_op_definition,
    OpAttr,
    AnyAttr,
    Operand,
    irdl_attr_definition,
    OptOpAttr,
    IRDLOperation,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))

_FloatTypeT = TypeVar("_FloatTypeT", bound=AnyFloat)


class FastMathFlag(Enum):
    REASSOC = "reassoc"
    NO_NANS = "nnan"
    NO_INFS = "ninf"
    NO_SIGNED_ZEROS = "nsz"
    ALLOW_RECIP = "arcp"
    ALLOW_CONTRACT = "contract"
    APPROX_FUNC = "afn"


@dataclass
class FastMathFlags:
    flags: Set[FastMathFlag]

    # TODO should we implement all/more set operators?
    def __or__(self, other: FastMathFlags):
        return FastMathFlags(self.flags | other.flags)

    def __contains__(self, item: FastMathFlag):
        return item in self.flags

    @staticmethod
    def try_parse(parser: Parser) -> Optional[FastMathFlags]:
        if parser.try_parse_characters("none") is not None:
            return FastMathFlags(set())
        if parser.try_parse_characters("fast") is not None:
            return FastMathFlags(set(FastMathFlag))

        for option in FastMathFlag:
            if parser.try_parse_characters(option.value) is not None:
                return FastMathFlags({option})

        return None


@irdl_attr_definition
class FastMathFlagsAttr(Data[FastMathFlags]):
    name = "arith.fastmath"

    @staticmethod
    def parse_parameter(parser: Parser) -> FastMathFlags:
        flag = FastMathFlags.try_parse(parser)
        if flag is None:
            return FastMathFlags(set())

        flags = [flag]
        while parser.parse_optional_punctuation(",") is not None:
            flag = parser.expect(
                lambda: FastMathFlags.try_parse(parser), "fastmath flag expected"
            )
            flags.append(flag)

        result = functools.reduce(FastMathFlags.__or__, flags, FastMathFlags(set()))
        return result

    def print_parameter(self, printer: Printer):
        data = self.data
        if len(data.flags) == 0:
            printer.print("none")
        elif len(data.flags) == len(FastMathFlag):
            printer.print("fast")
        else:
            # make sure we emit flags in a consistent order
            printer.print(",".join(flag.value for flag in FastMathFlag if flag in data))

    @staticmethod
    def from_flags(flags: FastMathFlags):
        return FastMathFlagsAttr(flags)


@irdl_op_definition
class Constant(IRDLOperation):
    name = "arith.constant"
    result: Annotated[OpResult, AnyAttr()]
    value: OpAttr[Attribute]

    @staticmethod
    def from_attr(attr: Attribute, typ: Attribute) -> Constant:
        return Constant.create(result_types=[typ], attributes={"value": attr})

    @staticmethod
    def from_int_and_width(
        val: int | IntAttr, typ: int | IntegerType | IndexType
    ) -> Constant:
        if isinstance(typ, int):
            typ = IntegerType(typ)
        return Constant.create(
            result_types=[typ], attributes={"value": IntegerAttr(val, typ)}
        )

    # To add tests for this constructor
    @staticmethod
    def from_float_and_width(
        val: float | FloatAttr[_FloatTypeT], typ: _FloatTypeT
    ) -> Constant:
        if isinstance(val, float):
            val = FloatAttr(val, typ)
        return Constant.create(result_types=[typ], attributes={"value": val})


_T = TypeVar("_T", bound=Attribute)


class BinaryOperation(IRDLOperation, Generic[_T]):
    """A generic base class for arith's binary operation.

    They all have two operands and one result of a same type."""

    T = Annotated[Attribute, ConstraintVar("T"), _T]

    lhs: Annotated[Operand, T]
    rhs: Annotated[Operand, T]
    result: Annotated[OpResult, T]

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).typ
        super().__init__(operands=[operand1, operand2], result_types=[result_type])

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_operand()
        parser.parse_punctuation(",")
        rhs = parser.parse_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        return cls(lhs, rhs, result_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.result.typ)

    def __hash__(self) -> int:
        return id(self)


SignlessIntegerBinaryOp = BinaryOperation[Annotated[Attribute, signlessIntegerLike]]
FloatingPointLikeBinaryOp = BinaryOperation[Annotated[Attribute, floatingPointLike]]
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
class RemSI(IntegerBinaryOp):
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
class ShLI(IntegerBinaryOp):
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
class ShRSI(IntegerBinaryOp):
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
        if operand1.typ != operand2.typ:
            raise TypeError(
                f"Comparison operands must have same type, but "
                f"provided {operand1.typ} and {operand2.typ}"
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
    predicate: OpAttr[AnyIntegerAttr]
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType(1)]

    @staticmethod
    def get(
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        arg: int | str,
    ) -> Cmpi:
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

        return Cmpi.build(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            attributes={"predicate": IntegerAttr.from_int_and_width(arg, 64)},
        )


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
    predicate: OpAttr[AnyIntegerAttr]
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, IntegerType(1)]

    @staticmethod
    def get(
        operand1: SSAValue | Operation, operand2: SSAValue | Operation, arg: int | str
    ) -> Cmpf:
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

        return Cmpf.build(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            attributes={"predicate": IntegerAttr.from_int_and_width(arg, 64)},
        )


@irdl_op_definition
class Select(IRDLOperation):
    """
    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.
    """

    name = "arith.select"
    cond: Annotated[Operand, IntegerType(1)]  # should be unsigned
    lhs: Annotated[Operand, Attribute]
    rhs: Annotated[Operand, Attribute]
    result: Annotated[OpResult, Attribute]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.cond.typ != IntegerType(1):
            raise VerifyException("Condition has to be of type !i1")
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException("expect all input and output types to be equal")

    @staticmethod
    def get(
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        operand3: Operation | SSAValue,
    ) -> Select:
        operand2 = SSAValue.get(operand2)
        return Select.build(
            operands=[operand1, operand2, operand3], result_types=[operand2.typ]
        )


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
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ) -> Negf:
        operand = SSAValue.get(operand)
        return Negf.build(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.typ],
        )


@irdl_op_definition
class Maxf(FloatingPointLikeBinaryOp):
    name = "arith.maxf"


@irdl_op_definition
class Minf(FloatingPointLikeBinaryOp):
    name = "arith.minf"


@irdl_op_definition
class IndexCastOp(IRDLOperation):
    name = "arith.index_cast"

    input: Operand

    result: OpResult

    @staticmethod
    def get(input_arg: SSAValue | Operation, target_type: Attribute):
        return IndexCastOp.build(operands=[input_arg], result_types=[target_type])


@irdl_op_definition
class FPToSIOp(IRDLOperation):
    name = "arith.fptosi"

    input: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(op: SSAValue | Operation, target_typ: IntegerType):
        return FPToSIOp.build(operands=[op], result_types=[target_typ])


@irdl_op_definition
class SIToFPOp(IRDLOperation):
    name = "arith.sitofp"

    input: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(op: SSAValue | Operation, target_typ: AnyFloat):
        return SIToFPOp.build(operands=[op], result_types=[target_typ])


@irdl_op_definition
class ExtFOp(IRDLOperation):
    name = "arith.extf"

    input: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(op: SSAValue | Operation, target_typ: AnyFloat):
        return ExtFOp.build(operands=[op], result_types=[target_typ])


@irdl_op_definition
class TruncFOp(IRDLOperation):
    name = "arith.truncf"

    input: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(op: SSAValue | Operation, target_typ: AnyFloat):
        return ExtFOp.build(operands=[op], result_types=[target_typ])


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
    ],
    [
        FastMathFlagsAttr,
    ],
)
