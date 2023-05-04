from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, TypeVar, Union, Set, Optional

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
    name: str = "arith.fastmath"

    @staticmethod
    def parse_parameter(parser: Parser) -> FastMathFlags:
        flags = parser.parse_list_of(
            lambda: FastMathFlags.try_parse(parser), "Expected fast math flags"
        )
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
    name: str = "arith.constant"
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
            result_types=[typ], attributes={"value": IntegerAttr.from_params(val, typ)}
        )

    # To add tests for this constructor
    @staticmethod
    def from_float_and_width(
        val: float | FloatAttr[_FloatTypeT], typ: _FloatTypeT
    ) -> Constant:
        if isinstance(val, float):
            val = FloatAttr(val, typ)
        return Constant.create(result_types=[typ], attributes={"value": val})


class BinaryOperation(IRDLOperation):
    """A generic operation. Operation definitions inherit this class."""

    # TODO replace with trait
    def verify_(self) -> None:
        if len(self.operands) != 2 or len(self.results) != 1:
            raise VerifyException("Binary operation expects 2 operands and 1 result.")
        if not (self.operands[0].typ == self.operands[1].typ == self.results[0].typ):
            raise VerifyException("expect all input and result types to be equal")

    def __hash__(self) -> int:
        return id(self)


@irdl_op_definition
class Addi(BinaryOperation):
    name: str = "arith.addi"

    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    traits = frozenset([Pure()])

    def __init__(
        self,
        operand1: Union[Operation, SSAValue],
        operand2: Union[Operation, SSAValue],
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).typ
        super().__init__(operands=[operand1, operand2], result_types=[result_type])


@irdl_op_definition
class Muli(BinaryOperation):
    name: str = "arith.muli"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Muli:
        operand1 = SSAValue.get(operand1)
        return Muli.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class Subi(BinaryOperation):
    name: str = "arith.subi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Subi:
        operand1 = SSAValue.get(operand1)
        return Subi.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class DivUI(BinaryOperation):
    """
    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.
    """

    name: str = "arith.divui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> DivUI:
        operand1 = SSAValue.get(operand1)
        return DivUI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class DivSI(BinaryOperation):
    """
    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.
    """

    name: str = "arith.divsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> DivSI:
        operand1 = SSAValue.get(operand1)
        return DivSI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class FloorDivSI(BinaryOperation):
    """
    Signed floor integer division. Rounds towards negative infinity i.e. `5 / -2 = -3`.
    """

    name: str = "arith.floordivsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> FloorDivSI:
        operand1 = SSAValue.get(operand1)
        return FloorDivSI.build(
            operands=[operand1, operand2], result_types=[operand1.typ]
        )


@irdl_op_definition
class CeilDivSI(BinaryOperation):
    name: str = "arith.ceildivsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> CeilDivSI:
        operand1 = SSAValue.get(operand1)
        return CeilDivSI.build(
            operands=[operand1, operand2], result_types=[operand1.typ]
        )


@irdl_op_definition
class CeilDivUI(BinaryOperation):
    name: str = "arith.ceildivui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> CeilDivUI:
        operand1 = SSAValue.get(operand1)
        return CeilDivUI.build(
            operands=[operand1, operand2], result_types=[operand1.typ]
        )


@irdl_op_definition
class RemUI(BinaryOperation):
    name: str = "arith.remui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> RemUI:
        operand1 = SSAValue.get(operand1)
        return RemUI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class RemSI(BinaryOperation):
    name: str = "arith.remsi"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> RemSI:
        operand1 = SSAValue.get(operand1)
        return RemSI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class MinUI(BinaryOperation):
    name: str = "arith.minui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> MinUI:
        operand1 = SSAValue.get(operand1)
        return MinUI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class MaxUI(BinaryOperation):
    name: str = "arith.maxui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> MaxUI:
        operand1 = SSAValue.get(operand1)
        return MaxUI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class MinSI(BinaryOperation):
    name: str = "arith.minsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> MinSI:
        operand1 = SSAValue.get(operand1)
        return MinSI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class MaxSI(BinaryOperation):
    name: str = "arith.maxsi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> MaxSI:
        operand1 = SSAValue.get(operand1)
        return MaxSI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class AndI(BinaryOperation):
    name: str = "arith.andi"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> AndI:
        operand1 = SSAValue.get(operand1)
        return AndI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class OrI(BinaryOperation):
    name: str = "arith.ori"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> OrI:
        operand1 = SSAValue.get(operand1)
        return OrI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class XOrI(BinaryOperation):
    name: str = "arith.xori"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> XOrI:
        operand1 = SSAValue.get(operand1)
        return XOrI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class ShLI(IRDLOperation):
    """
    The `shli` operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.
    """

    name: str = "arith.shli"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException("expect all input and output types to be equal")

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> ShLI:
        operand1 = SSAValue.get(operand1)
        return ShLI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class ShRUI(IRDLOperation):
    """
    The `shrui` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as unsigned. The high order bits are
    always filled with zeros.
    """

    name: str = "arith.shrui"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException("expect all input and output types to be equal")

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> ShRUI:
        operand1 = SSAValue.get(operand1)
        return ShRUI.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class ShRSI(IRDLOperation):
    """
    The `shrsi` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as signed. The high order bits in the
    output are filled with copies of the most-significant bit of the shifted
    value (which means that the sign of the value is preserved).
    """

    name: str = "arith.shrsi"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    # TODO replace with trait
    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ or self.rhs.typ != self.result.typ:
            raise VerifyException("expect all input and output types to be equal")

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> ShRSI:
        operand1 = SSAValue.get(operand1)
        return ShRSI.build(operands=[operand1, operand2], result_types=[operand1.typ])


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

    name: str = "arith.cmpi"
    predicate: OpAttr[AnyIntegerAttr]
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType(1)]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue],
        operand2: Union[Operation, SSAValue],
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

    name: str = "arith.cmpf"
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

    name: str = "arith.select"
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
        operand1: Union[Operation, SSAValue],
        operand2: Union[Operation, SSAValue],
        operand3: Union[Operation, SSAValue],
    ) -> Select:
        operand2 = SSAValue.get(operand2)
        return Select.build(
            operands=[operand1, operand2, operand3], result_types=[operand2.typ]
        )


@irdl_op_definition
class Addf(BinaryOperation):
    name: str = "arith.addf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Addf:
        operand1 = SSAValue.get(operand1)
        return Addf.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class Subf(BinaryOperation):
    name: str = "arith.subf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Subf:
        operand1 = SSAValue.get(operand1)
        return Subf.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class Mulf(BinaryOperation):
    name: str = "arith.mulf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Mulf:
        operand1 = SSAValue.get(operand1)
        return Mulf.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class Divf(BinaryOperation):
    name: str = "arith.divf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Divf:
        operand1 = SSAValue.get(operand1)
        return Divf.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class Negf(IRDLOperation):
    name: str = "arith.negf"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> Negf:
        operand = SSAValue.get(operand)
        return Negf.build(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.typ],
        )


@irdl_op_definition
class Maxf(BinaryOperation):
    name: str = "arith.maxf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Maxf:
        operand1 = SSAValue.get(operand1)
        return Maxf.build(operands=[operand1, operand2], result_types=[operand1.typ])


@irdl_op_definition
class Minf(BinaryOperation):
    name: str = "arith.minf"
    lhs: Annotated[Operand, floatingPointLike]
    rhs: Annotated[Operand, floatingPointLike]
    result: Annotated[OpResult, floatingPointLike]

    @staticmethod
    def get(
        operand1: Union[Operation, SSAValue], operand2: Union[Operation, SSAValue]
    ) -> Minf:
        operand1 = SSAValue.get(operand1)
        return Minf.build(operands=[operand1, operand2], result_types=[operand1.typ])


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
