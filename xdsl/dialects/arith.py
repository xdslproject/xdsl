from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal, TypeVar, cast, overload

from xdsl.dialects.builtin import (
    AnyFloat,
    AnyFloatConstr,
    AnyIntegerAttr,
    AnyIntegerAttrConstr,
    ContainerOf,
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    TensorType,
    UnrankedTensorType,
    VectorType,
)
from xdsl.dialects.utils import FastMathAttrBase, FastMathFlag
from xdsl.ir import Attribute, BitEnumAttribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    BaseAttr,
    IRDLOperation,
    TypedAttributeConstraint,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    ConditionallySpeculatable,
    ConstantLike,
    HasCanonicalizationPatternsTrait,
    NoMemoryEffect,
    Pure,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

boolLike = ContainerOf(IntegerType(1))
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


@irdl_attr_definition
class FastMathFlagsAttr(FastMathAttrBase):
    """
    arith.fastmath is a mirror of LLVMs fastmath flags.
    """

    name = "arith.fastmath"

    def __init__(self, flags: None | Sequence[FastMathFlag] | Literal["none", "fast"]):
        # irdl_attr_definition defines an __init__ if none is defined, so we need to
        # explicitely define one here.
        super().__init__(flags)


class IntegerOverflowFlag(StrEnum):
    NSW = "nsw"
    NUW = "nuw"


@irdl_attr_definition
class IntegerOverflowAttr(BitEnumAttribute[IntegerOverflowFlag]):
    name = "arith.overflow"

    none_value = "none"

    def __init__(self, flags: None | Sequence[IntegerOverflowFlag] | Literal["none"]):
        # irdl_attr_definition defines an __init__ if none is defined, so we need to
        # explicitely define one here.
        super().__init__(flags)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "arith.constant"
    _T: ClassVar = VarConstraint("T", AnyAttr())
    result = result_def(_T)
    value = prop_def(
        TypedAttributeConstraint(
            AnyIntegerAttrConstr
            | BaseAttr[FloatAttr[AnyFloat]](FloatAttr)
            | BaseAttr(DenseIntOrFPElementsAttr),
            _T,
        )
    )

    traits = traits_def(ConstantLike(), Pure())

    assembly_format = "attr-dict $value"

    @overload
    def __init__(
        self,
        value: AnyIntegerAttr | FloatAttr[AnyFloat] | DenseIntOrFPElementsAttr,
        value_type: None = None,
    ) -> None: ...

    @overload
    def __init__(self, value: Attribute, value_type: Attribute) -> None: ...

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
    def from_int_and_width(
        value: int | IntAttr, value_type: int | IntegerType | IndexType
    ) -> ConstantOp:
        if isinstance(value_type, int):
            value_type = IntegerType(value_type)
        return ConstantOp.create(
            result_types=[value_type],
            properties={"value": IntegerAttr(value, value_type)},
        )


_T = TypeVar("_T", bound=Attribute)


class SignlessIntegerBinaryOperation(IRDLOperation, abc.ABC):
    """A generic base class for arith's binary operations on signless integers."""

    T: ClassVar = VarConstraint("T", signlessIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(operands=[operand1, operand2], result_types=[result_type])

    def __hash__(self) -> int:
        return id(self)


class SignlessIntegerBinaryOperationWithOverflow(
    SignlessIntegerBinaryOperation, abc.ABC
):
    """
    A generic base class for arith's binary operations on signless integers which
    can overflow.
    """

    overflow_flags = prop_def(
        IntegerOverflowAttr,
        default_value=IntegerOverflowAttr("none"),
        prop_name="overflowFlags",
    )

    assembly_format = (
        "$lhs `,` $rhs (`overflow` `` $overflowFlags^)? attr-dict `:` type($result)"
    )

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        result_type: Attribute | None = None,
        overflow: IntegerOverflowAttr = IntegerOverflowAttr("none"),
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        IRDLOperation.__init__(
            self,
            operands=[operand1, operand2],
            properties={"overflowFlags": overflow},
            result_types=[result_type],
        )


class FloatingPointLikeBinaryOpHasCanonicalizationPatternsTrait(
    HasCanonicalizationPatternsTrait
):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.arith import FoldConstConstOp

        return (FoldConstConstOp(),)


class FloatingPointLikeBinaryOpHasFastReassociativeCanonicalizationPatternsTrait(
    HasCanonicalizationPatternsTrait
):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.arith import (
            FoldConstConstOp,
            FoldConstsByReassociation,
        )

        return FoldConstsByReassociation(), FoldConstConstOp()


class FloatingPointLikeBinaryOperation(IRDLOperation, abc.ABC):
    """A generic base class for arith's binary operations on floats."""

    T: ClassVar = VarConstraint("T", floatingPointLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    fastmath = opt_prop_def(FastMathFlagsAttr)

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        flags: FastMathFlagsAttr | None = None,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(
            operands=[operand1, operand2],
            result_types=[result_type],
            properties={"fastmath": flags},
        )

    @classmethod
    def parse(cls, parser: Parser):
        lhs = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        rhs = parser.parse_unresolved_operand()
        flags = FastMathFlagsAttr("none")
        if parser.parse_optional_keyword("fastmath") is not None:
            flags = FastMathFlagsAttr(FastMathFlagsAttr.parse_parameter(parser))
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        (lhs, rhs) = parser.resolve_operands([lhs, rhs], 2 * [result_type], parser.pos)
        return cls(lhs, rhs, flags, result_type)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)
        if self.fastmath is not None and self.fastmath != FastMathFlagsAttr("none"):
            printer.print(" fastmath")
            self.fastmath.print_parameter(printer)
        printer.print(" : ")
        printer.print_attribute(self.result.type)


class AddiOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.arith import AddImmediateZero

        return (AddImmediateZero(),)


@irdl_op_definition
class AddiOp(SignlessIntegerBinaryOperationWithOverflow):
    name = "arith.addi"

    traits = traits_def(Pure(), AddiOpHasCanonicalizationPatternsTrait())


@irdl_op_definition
class AddUIExtendedOp(IRDLOperation):
    """
    An add operation on an unsigned representation of integers that returns a flag
    indicating if the result overflowed.
    """

    name = "arith.addui_extended"

    traits = traits_def(Pure())

    T: ClassVar = VarConstraint("T", signlessIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)

    sum = result_def(T)
    overflow = result_def(boolLike)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)"

    traits = traits_def(Pure())

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        attributes: Mapping[str, Attribute] | None = None,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        overflow_type = AddUIExtendedOp.infer_overflow_type(result_type)
        super().__init__(
            operands=[operand1, operand2],
            result_types=[result_type, overflow_type],
            attributes=attributes,
        )

    def verify_(self):
        expected_overflow_type = AddUIExtendedOp.infer_overflow_type(self.lhs.type)
        if self.overflow.type != expected_overflow_type:
            raise VerifyException(
                f"overflow type {self.overflow.type} does not "
                f"match input types {self.lhs.type}. Expected {expected_overflow_type}"
            )

    @staticmethod
    def infer_overflow_type(input_type: Attribute) -> Attribute:
        if isinstance(input_type, IntegerType):
            return IntegerType(1)
        if isinstance(input_type, VectorType):
            return VectorType(
                IntegerType(1), input_type.shape, input_type.num_scalable_dims
            )
        if isinstance(input_type, UnrankedTensorType):
            return UnrankedTensorType(IntegerType(1))
        if isinstance(input_type, TensorType):
            return TensorType(IntegerType(1), input_type.shape, input_type.encoding)
        raise ValueError(
            f"Unsupported input type for {AddUIExtendedOp.name}: {input_type}"
        )


@irdl_op_definition
class MuliOp(SignlessIntegerBinaryOperationWithOverflow):
    name = "arith.muli"

    traits = traits_def(Pure())


class MulExtendedBase(IRDLOperation):
    """Base class for extended multiplication operations."""

    T: ClassVar = VarConstraint("T", signlessIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    low = result_def(T)
    high = result_def(T)

    traits = traits_def(Pure())

    def __init__(
        self,
        operand1: SSAValue,
        operand2: SSAValue,
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            result_type = SSAValue.get(operand1).type
        super().__init__(
            operands=[operand1, operand2], result_types=[result_type, result_type]
        )

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"


@irdl_op_definition
class MulUIExtendedOp(MulExtendedBase):
    """Extended unsigned integer multiplication operation."""

    name = "arith.mului_extended"


@irdl_op_definition
class MulSIExtendedOp(MulExtendedBase):
    """Extended unsigned integer multiplication operation."""

    name = "arith.mulsi_extended"


@irdl_op_definition
class SubiOp(SignlessIntegerBinaryOperationWithOverflow):
    name = "arith.subi"

    traits = traits_def(Pure())


class DivUISpeculatable(ConditionallySpeculatable):
    @classmethod
    def is_speculatable(cls, op: Operation):
        op = cast(DivUIOp, op)
        if not isinstance(cst := op.rhs.owner, ConstantOp):
            return False
        value = cast(IntegerAttr[IntegerType | IndexType], cst.value)
        return value.value.data != 0


@irdl_op_definition
class DivUIOp(SignlessIntegerBinaryOperation):
    """
    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.
    """

    name = "arith.divui"

    traits = traits_def(NoMemoryEffect(), DivUISpeculatable())


@irdl_op_definition
class DivSIOp(SignlessIntegerBinaryOperation):
    """
    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.
    """

    name = "arith.divsi"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class FloorDivSIOp(SignlessIntegerBinaryOperation):
    """
    Signed floor integer division. Rounds towards negative infinity i.e. `5 / -2 = -3`.
    """

    name = "arith.floordivsi"

    traits = traits_def(Pure())


@irdl_op_definition
class CeilDivSIOp(SignlessIntegerBinaryOperation):
    name = "arith.ceildivsi"

    traits = traits_def(Pure())


@irdl_op_definition
class CeilDivUIOp(SignlessIntegerBinaryOperation):
    name = "arith.ceildivui"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class RemUIOp(SignlessIntegerBinaryOperation):
    name = "arith.remui"


@irdl_op_definition
class RemSIOp(SignlessIntegerBinaryOperation):
    name = "arith.remsi"

    traits = traits_def(Pure())


@irdl_op_definition
class MinUIOp(SignlessIntegerBinaryOperation):
    name = "arith.minui"

    traits = traits_def(Pure())


@irdl_op_definition
class MaxUIOp(SignlessIntegerBinaryOperation):
    name = "arith.maxui"

    traits = traits_def(Pure())


@irdl_op_definition
class MinSIOp(SignlessIntegerBinaryOperation):
    name = "arith.minsi"

    traits = traits_def(Pure())


@irdl_op_definition
class MaxSIOp(SignlessIntegerBinaryOperation):
    name = "arith.maxsi"

    traits = traits_def(Pure())


@irdl_op_definition
class AndIOp(SignlessIntegerBinaryOperation):
    name = "arith.andi"

    traits = traits_def(Pure())


@irdl_op_definition
class OrIOp(SignlessIntegerBinaryOperation):
    name = "arith.ori"

    traits = traits_def(Pure())


@irdl_op_definition
class XOrIOp(SignlessIntegerBinaryOperation):
    name = "arith.xori"

    traits = traits_def(Pure())


@irdl_op_definition
class ShLIOp(SignlessIntegerBinaryOperationWithOverflow):
    """
    The `shli` operation shifts an integer value to the left by a variable
    amount. The low order bits are filled with zeros.
    """

    name = "arith.shli"

    traits = traits_def(Pure())


@irdl_op_definition
class ShRUIOp(SignlessIntegerBinaryOperation):
    """
    The `shrui` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as unsigned. The high order bits are
    always filled with zeros.
    """

    name = "arith.shrui"

    traits = traits_def(Pure())


@irdl_op_definition
class ShRSIOp(SignlessIntegerBinaryOperation):
    """
    The `shrsi` operation shifts an integer value to the right by a variable
    amount. The integer is interpreted as signed. The high order bits in the
    output are filled with copies of the most-significant bit of the shifted
    value (which means that the sign of the value is preserved).
    """

    name = "arith.shrsi"

    traits = traits_def(Pure())


class ComparisonOperation(IRDLOperation):
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

    traits = traits_def(Pure())


@irdl_op_definition
class CmpiOp(ComparisonOperation):
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
    predicate = prop_def(AnyIntegerAttr)
    lhs = operand_def(signlessIntegerLike)
    rhs = operand_def(signlessIntegerLike)
    result = result_def(IntegerType(1))

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        arg: int | str,
    ):
        operand1 = SSAValue.get(operand1)
        operand2 = SSAValue.get(operand2)
        CmpiOp._validate_operand_types(operand1, operand2)

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
            arg = CmpiOp._get_comparison_predicate(arg, cmpi_comparison_operations)

        super().__init__(
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
class CmpfOp(ComparisonOperation):
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
    predicate = prop_def(AnyIntegerAttr)
    lhs = operand_def(floatingPointLike)
    rhs = operand_def(floatingPointLike)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))
    result = result_def(IntegerType(1))

    def __init__(
        self,
        operand1: SSAValue | Operation,
        operand2: SSAValue | Operation,
        arg: int | str,
        fastmath: FastMathFlagsAttr = FastMathFlagsAttr("none"),
    ):
        operand1 = SSAValue.get(operand1)
        operand2 = SSAValue.get(operand2)

        CmpfOp._validate_operand_types(operand1, operand2)

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
            arg = CmpfOp._get_comparison_predicate(arg, cmpf_comparison_operations)

        super().__init__(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            properties={
                "predicate": IntegerAttr.from_int_and_width(arg, 64),
                "fastmath": fastmath,
            },
        )

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_identifier()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        if parser.parse_optional_keyword("fastmath"):
            fastmath = FastMathFlagsAttr(FastMathFlagsAttr.parse_parameter(parser))
        else:
            fastmath = FastMathFlagsAttr("none")
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], 2 * [input_type], parser.pos
        )

        return cls(operand1, operand2, arg, fastmath)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_string(CMPF_COMPARISON_OPERATIONS[self.predicate.value.data])
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        if self.fastmath != FastMathFlagsAttr("none"):
            printer.print_string(" fastmath")
            self.fastmath.print_parameter(printer)
        printer.print(" : ")
        printer.print_attribute(self.lhs.type)


class SelectHasCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.arith import (
            SelectConstPattern,
            SelectSamePattern,
            SelectTrueFalsePattern,
        )

        return (SelectConstPattern(), SelectTrueFalsePattern(), SelectSamePattern())


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.
    """

    name = "arith.select"
    cond = operand_def(IntegerType(1))  # should be unsigned
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(Attribute)

    traits = traits_def(Pure(), SelectHasCanonicalizationPatterns())

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
        super().__init__(
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
class AddfOp(FloatingPointLikeBinaryOperation):
    name = "arith.addf"

    traits = traits_def(
        Pure(),
        FloatingPointLikeBinaryOpHasFastReassociativeCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class SubfOp(FloatingPointLikeBinaryOperation):
    name = "arith.subf"

    traits = traits_def(
        Pure(), FloatingPointLikeBinaryOpHasCanonicalizationPatternsTrait()
    )


@irdl_op_definition
class MulfOp(FloatingPointLikeBinaryOperation):
    name = "arith.mulf"

    traits = traits_def(
        Pure(),
        FloatingPointLikeBinaryOpHasFastReassociativeCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class DivfOp(FloatingPointLikeBinaryOperation):
    name = "arith.divf"

    traits = traits_def(
        Pure(), FloatingPointLikeBinaryOpHasCanonicalizationPatternsTrait()
    )


@irdl_op_definition
class NegfOp(IRDLOperation):
    name = "arith.negf"
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(floatingPointLike)
    result = result_def(floatingPointLike)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        operand = SSAValue.get(operand)
        super().__init__(
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
class MaximumfOp(FloatingPointLikeBinaryOperation):
    """
    Returns the maximum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.
    """

    name = "arith.maximumf"

    traits = traits_def(Pure())


@irdl_op_definition
class MaxnumfOp(FloatingPointLikeBinaryOperation):
    """
    Returns the maximum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.
    """

    name = "arith.maxnumf"

    traits = traits_def(Pure())


@irdl_op_definition
class MinimumfOp(FloatingPointLikeBinaryOperation):
    """
    Returns the minimum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.
    """

    name = "arith.minimumf"

    traits = traits_def(Pure())


@irdl_op_definition
class MinnumfOp(FloatingPointLikeBinaryOperation):
    """
    Returns the minimum of the two arguments. If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.
    """

    name = "arith.minnumf"

    traits = traits_def(Pure())


@irdl_op_definition
class IndexCastOp(IRDLOperation):
    name = "arith.index_cast"

    input = operand_def(base(IntegerType) | base(IndexType))

    result = result_def(base(IntegerType) | base(IndexType))

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    def __init__(self, input_arg: SSAValue | Operation, target_type: Attribute):
        super().__init__(operands=[input_arg], result_types=[target_type])

    def verify_(self) -> None:
        it = IndexType
        # exactly one of input or result must be of IndexType, no more, no less.
        if not isinstance(self.input.type, it) ^ isinstance(self.result.type, it):
            raise VerifyException(
                f"'arith.index_cast' op operand type '{self.input.type}' and result type '{self.input.type}' are cast incompatible"
            )


@irdl_op_definition
class FPToSIOp(IRDLOperation):
    name = "arith.fptosi"

    input = operand_def(AnyFloatConstr)
    result = result_def(IntegerType)

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    traits = traits_def(Pure())

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        super().__init__(operands=[op], result_types=[target_type])


@irdl_op_definition
class SIToFPOp(IRDLOperation):
    name = "arith.sitofp"

    input = operand_def(IntegerType)
    result = result_def(AnyFloatConstr)

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    traits = traits_def(Pure())

    def __init__(self, op: SSAValue | Operation, target_type: AnyFloat):
        super().__init__(operands=[op], result_types=[target_type])


@irdl_op_definition
class ExtFOp(IRDLOperation):
    name = "arith.extf"

    input = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    def __init__(self, op: SSAValue | Operation, target_type: AnyFloat):
        super().__init__(operands=[op], result_types=[target_type])

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    traits = traits_def(Pure())


@irdl_op_definition
class TruncFOp(IRDLOperation):
    name = "arith.truncf"

    input = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    def __init__(self, op: SSAValue | Operation, target_type: AnyFloat):
        super().__init__(operands=[op], result_types=[target_type])

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    traits = traits_def(Pure())


@irdl_op_definition
class TruncIOp(IRDLOperation):
    name = "arith.trunci"

    input = operand_def(IntegerType)
    result = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        super().__init__(operands=[op], result_types=[target_type])

    def verify_(self) -> None:
        assert isinstance(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if not self.result.type.width.data < self.input.type.width.data:
            raise VerifyException(
                "Destination bit-width must be smaller than the input bit-width"
            )

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    traits = traits_def(Pure())


@irdl_op_definition
class ExtSIOp(IRDLOperation):
    name = "arith.extsi"

    input = operand_def(IntegerType)
    result = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        super().__init__(operands=[op], result_types=[target_type])

    def verify_(self) -> None:
        assert isinstance(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if not self.result.type.width.data > self.input.type.width.data:
            raise VerifyException(
                "Destination bit-width must be larger than the input bit-width"
            )

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"


@irdl_op_definition
class ExtUIOp(IRDLOperation):
    name = "arith.extui"

    input = operand_def(IntegerType)
    result = result_def(IntegerType)

    def __init__(self, op: SSAValue | Operation, target_type: IntegerType):
        super().__init__(operands=[op], result_types=[target_type])

    def verify_(self) -> None:
        assert isinstance(self.input.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if not self.result.type.width.data > self.input.type.width.data:
            raise VerifyException(
                "Destination bit-width must be larger than the input bit-width"
            )

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    traits = traits_def(Pure())


Arith = Dialect(
    "arith",
    [
        ConstantOp,
        # Integer-like
        AddiOp,
        AddUIExtendedOp,
        SubiOp,
        MuliOp,
        MulUIExtendedOp,
        MulSIExtendedOp,
        DivUIOp,
        DivSIOp,
        FloorDivSIOp,
        CeilDivSIOp,
        CeilDivUIOp,
        RemUIOp,
        RemSIOp,
        MinSIOp,
        MaxSIOp,
        MinUIOp,
        MaxUIOp,
        # Float-like
        AddfOp,
        SubfOp,
        MulfOp,
        DivfOp,
        NegfOp,
        # Comparison/Condition
        CmpiOp,
        CmpfOp,
        SelectOp,
        # Logical
        AndIOp,
        OrIOp,
        XOrIOp,
        # Shift
        ShLIOp,
        ShRUIOp,
        ShRSIOp,
        # Min/Max
        MinimumfOp,
        MinnumfOp,
        MaximumfOp,
        MaxnumfOp,
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
        IntegerOverflowAttr,
    ],
)
