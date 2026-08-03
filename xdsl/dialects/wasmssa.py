from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, Generic, TypeAlias, cast

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I32,
    I64,
    I128,
    FlatSymbolRefAttrConstr,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    SymbolRefAttr,
    i32,
    i64,
)
from xdsl.ir import (
    Dialect,
    OpaqueSyntaxAttribute,
    Operation,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    ParamAttrConstraint,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Commutative, ConstantLike, NoMemoryEffect, Pure
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class FuncRefType(ParametrizedAttribute, TypeAttribute):
    """
    Opaque type for function reference
    """

    name = "wasmssa.funcref"


@irdl_attr_definition
class ExternRefType(ParametrizedAttribute, TypeAttribute):
    """
    Opaque type for extern reference
    """

    name = "wasmssa.externref"


RefType: TypeAlias = FuncRefType | ExternRefType
"""Type alias for opaque references in WebAssembly"""
WasmIntegerType: TypeAlias = I32 | I64
"""Type alias for integer types that are supported by WebAssembly"""
WasmFPType: TypeAlias = Float32Type | Float64Type
"""Type alias for floating-point types that are supported by WebAssembly"""
NumericType: TypeAlias = WasmIntegerType | WasmFPType
"""Type alias for numeric types that are supported by WebAssembly"""
ValType: TypeAlias = I128 | NumericType | FuncRefType | ExternRefType
"""Type alias for value types that are supported by WebAssembly"""

_NumericTypeInvT = TypeVar("_NumericTypeInvT", bound=NumericType, default=NumericType)
_NumericResultTypeInvT = TypeVar(
    "_NumericResultTypeInvT", bound=NumericType, default=NumericType
)


@irdl_attr_definition
class LimitType(ParametrizedAttribute, OpaqueSyntaxAttribute, TypeAttribute):
    """
    Wasm limit type

    Prints as `!wasmssa<limit[$min: $max]>`
    """

    name = "wasmssa.limit"

    min: IntAttr
    max: IntAttr | NoneAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[IntAttr, IntAttr | NoneAttr]:
        with parser.in_square_brackets():
            min = parser.parse_integer(False, False)
            parser.parse_punctuation(":")
            max = parser.parse_optional_integer(False, False)
        return (IntAttr(min), IntAttr(max) if max is not None else NoneAttr())

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_square_brackets():
            printer.print_int(self.min.data)
            printer.print_string(":")
            if not isinstance(self.max, NoneAttr):
                printer.print_string(" ")
                printer.print_int(self.max.data)


@irdl_attr_definition
class LocalRefType(ParametrizedAttribute, SpacedOpaqueSyntaxAttribute, TypeAttribute):
    """
    Type of a local variable

    Prints as `!wasmssa<local ref to $elementType>`
    """

    name = "wasmssa.local"

    elementType: ValType

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[TypeAttribute]:
        parser.parse_keyword("ref")
        parser.parse_keyword("to")
        ty = parser.parse_type()
        return [ty]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("ref to ")
        printer.print_attribute(self.elementType)


@irdl_attr_definition
class TableType(ParametrizedAttribute, SpacedOpaqueSyntaxAttribute, TypeAttribute):
    """
    Wasm table type

    Prints as `!wasmssa<tabletype $reference [$limit.min: $limit.max]>`
    """

    name = "wasmssa.tabletype"

    reference: RefType
    limit: LimitType

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[RefType, LimitType]:
        reference = cast(RefType, parser.parse_type())
        min, max = LimitType.parse_parameters(parser)

        return (reference, LimitType(min, max))

    def print_parameters(self, printer: Printer) -> None:
        printer.print_attribute(self.reference)
        printer.print_string(" ")
        self.limit.print_parameters(printer)


@irdl_op_definition
class ConstOp(IRDLOperation):
    """Define a WebAssembly numeric constant."""

    name = "wasmssa.const"

    T: ClassVar = VarConstraint.get("T", NumericType)

    value = prop_def(
        ParamAttrConstraint(IntegerAttr, (AnyAttr(), T))
        | ParamAttrConstraint(FloatAttr, (AnyAttr(), T))
    )
    result = result_def(T)

    traits = traits_def(ConstantLike())

    assembly_format = "$value attr-dict"

    def __init__(self, value: IntegerAttr | FloatAttr):
        super().__init__(
            properties={"value": value},
            result_types=[value.get_type()],
        )


@irdl_op_definition
class GlobalGetOp(IRDLOperation):
    """Return the value of a WebAssembly global."""

    name = "wasmssa.global_get"

    global_ = prop_def(FlatSymbolRefAttrConstr, prop_name="global")
    global_val = result_def(ValType)

    traits = traits_def(ConstantLike())

    assembly_format = "$global attr-dict `:` type($global_val)"

    def __init__(
        self,
        global_: str | SymbolRefAttr,
        result_type: ValType,
    ):
        super().__init__(
            properties={"global": SymbolRefAttr.get(global_)},
            result_types=[result_type],
        )


class BinaryNumericalOperation(IRDLOperation, ABC, Generic[_NumericTypeInvT]):
    """Base class for binary WebAssembly numeric operations."""

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(_NumericTypeInvT, allow_type_var=True)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    assembly_format = "$lhs $rhs `:` type($lhs) attr-dict"

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        lhs = SSAValue.get(lhs)
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class AddOp(BinaryNumericalOperation):
    """Sum two WebAssembly numeric values."""

    name = "wasmssa.add"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class AndOp(BinaryNumericalOperation):
    """Compute the bitwise AND between two values."""

    name = "wasmssa.and"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class DivOp(BinaryNumericalOperation[WasmFPType]):
    """Divide two floating-point values."""

    name = "wasmssa.div"

    traits = traits_def(Pure())


@irdl_op_definition
class DivUIOp(BinaryNumericalOperation[WasmIntegerType]):
    """Divide two values interpreted as unsigned integers."""

    name = "wasmssa.div_ui"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class DivSIOp(BinaryNumericalOperation[WasmIntegerType]):
    """Divide two values interpreted as signed integers."""

    name = "wasmssa.div_si"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class MulOp(BinaryNumericalOperation):
    """Multiply two values."""

    name = "wasmssa.mul"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class OrOp(BinaryNumericalOperation):
    """Compute the bitwise OR between two values."""

    name = "wasmssa.or"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class SubOp(BinaryNumericalOperation):
    """Subtract two values."""

    name = "wasmssa.sub"

    traits = traits_def(Pure())


@irdl_op_definition
class RemUIOp(BinaryNumericalOperation[WasmIntegerType]):
    """Compute the unsigned integer remainder of two values."""

    name = "wasmssa.rem_ui"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class RemSIOp(BinaryNumericalOperation[WasmIntegerType]):
    """Compute the signed integer remainder of two values."""

    name = "wasmssa.rem_si"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class XOrOp(BinaryNumericalOperation):
    """Compute the bitwise XOR between two values."""

    name = "wasmssa.xor"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class MinOp(BinaryNumericalOperation[WasmFPType]):
    """Compute the minimum of two floating-point values."""

    name = "wasmssa.min"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class MaxOp(BinaryNumericalOperation[WasmFPType]):
    """Compute the maximum of two floating-point values."""

    name = "wasmssa.max"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class CopySignOp(BinaryNumericalOperation[WasmFPType]):
    """Copy the sign of the second floating-point value to the first."""

    name = "wasmssa.copysign"

    traits = traits_def(Pure())


class BinaryComparisonOperation(IRDLOperation, ABC, Generic[_NumericTypeInvT]):
    """Base class for binary WebAssembly comparison operations."""

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(_NumericTypeInvT, allow_type_var=True)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(I32)

    assembly_format = "$lhs $rhs `:` type($lhs) `->` type($result) attr-dict"

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class EqOp(BinaryComparisonOperation):
    """Check if two numeric values are equal."""

    name = "wasmssa.eq"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class NeOp(BinaryComparisonOperation):
    """Check if two numeric values are different."""

    name = "wasmssa.ne"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class LtSIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if a signed integer is less than another."""

    name = "wasmssa.lt_si"

    traits = traits_def(Pure())


@irdl_op_definition
class LtUIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if an unsigned integer is less than another."""

    name = "wasmssa.lt_ui"

    traits = traits_def(Pure())


@irdl_op_definition
class LeSIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if a signed integer is less than or equal to another."""

    name = "wasmssa.le_si"

    traits = traits_def(Pure())


@irdl_op_definition
class LeUIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if an unsigned integer is less than or equal to another."""

    name = "wasmssa.le_ui"

    traits = traits_def(Pure())


@irdl_op_definition
class GtSIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if a signed integer is greater than another."""

    name = "wasmssa.gt_si"

    traits = traits_def(Pure())


@irdl_op_definition
class GtUIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if an unsigned integer is greater than another."""

    name = "wasmssa.gt_ui"

    traits = traits_def(Pure())


@irdl_op_definition
class GeSIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if a signed integer is greater than or equal to another."""

    name = "wasmssa.ge_si"

    traits = traits_def(Pure())


@irdl_op_definition
class GeUIOp(BinaryComparisonOperation[WasmIntegerType]):
    """Check if an unsigned integer is greater than or equal to another."""

    name = "wasmssa.ge_ui"

    traits = traits_def(Pure())


@irdl_op_definition
class LtOp(BinaryComparisonOperation[WasmFPType]):
    """Check if a floating-point value is less than another."""

    name = "wasmssa.lt"

    traits = traits_def(Pure())


@irdl_op_definition
class LeOp(BinaryComparisonOperation[WasmFPType]):
    """Check if a floating-point value is less than or equal to another."""

    name = "wasmssa.le"

    traits = traits_def(Pure())


@irdl_op_definition
class GtOp(BinaryComparisonOperation[WasmFPType]):
    """Check if a floating-point value is greater than another."""

    name = "wasmssa.gt"

    traits = traits_def(Pure())


@irdl_op_definition
class GeOp(BinaryComparisonOperation[WasmFPType]):
    """Check if a floating-point value is greater than or equal to another."""

    name = "wasmssa.ge"

    traits = traits_def(Pure())


@irdl_op_definition
class EqzOp(IRDLOperation):
    """Check if an integer value is equal to zero."""

    name = "wasmssa.eqz"

    input = operand_def(WasmIntegerType)
    result = result_def(I32)

    traits = traits_def(Pure())

    assembly_format = "$input `:` type($input) `->` type($result) attr-dict"

    def __init__(self, input: SSAValue | Operation):
        super().__init__(operands=[input], result_types=[i32])


class ShiftRotateOperation(IRDLOperation, ABC):
    """Base class for WebAssembly integer shift and rotate operations."""

    T: ClassVar = VarConstraint.get("T", WasmIntegerType)

    val = operand_def(T)
    bits = operand_def(T)
    result = result_def(T)

    assembly_format = "$val `by` $bits `bits` `:` type($val) attr-dict"

    traits = traits_def(Pure())

    def __init__(
        self,
        val: SSAValue | Operation,
        bits: SSAValue | Operation,
    ):
        val = SSAValue.get(val)
        super().__init__(operands=[val, bits], result_types=[val.type])


@irdl_op_definition
class ShLOp(ShiftRotateOperation):
    """Shift an integer value left."""

    name = "wasmssa.shl"


@irdl_op_definition
class ShRSOp(ShiftRotateOperation):
    """Shift a signed integer value right."""

    name = "wasmssa.shr_s"


@irdl_op_definition
class ShRUOp(ShiftRotateOperation):
    """Shift an unsigned integer value right."""

    name = "wasmssa.shr_u"


@irdl_op_definition
class RotlOp(ShiftRotateOperation):
    """Rotate an integer value left."""

    name = "wasmssa.rotl"


@irdl_op_definition
class RotrOp(ShiftRotateOperation):
    """Rotate an integer value right."""

    name = "wasmssa.rotr"


class UnaryNumericalOperation(IRDLOperation, ABC, Generic[_NumericTypeInvT]):
    """Base class for unary WebAssembly numeric operations."""

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(_NumericTypeInvT, allow_type_var=True)
    )

    src = operand_def(T)
    result = result_def(T)

    assembly_format = "$src `:` type($src) attr-dict"

    traits = traits_def(Pure())

    def __init__(self, src: SSAValue | Operation):
        src = SSAValue.get(src)
        super().__init__(operands=[src], result_types=[src.type])


@irdl_op_definition
class AbsOp(UnaryNumericalOperation[WasmFPType]):
    """Compute the absolute value of a floating-point value."""

    name = "wasmssa.abs"


@irdl_op_definition
class CeilOp(UnaryNumericalOperation[WasmFPType]):
    """Round a floating-point value toward positive infinity."""

    name = "wasmssa.ceil"


@irdl_op_definition
class FloorOp(UnaryNumericalOperation[WasmFPType]):
    """Round a floating-point value toward negative infinity."""

    name = "wasmssa.floor"


@irdl_op_definition
class NegOp(UnaryNumericalOperation[WasmFPType]):
    """Negate a floating-point value."""

    name = "wasmssa.neg"


@irdl_op_definition
class SqrtOp(UnaryNumericalOperation[WasmFPType]):
    """Compute the square root of a floating-point value."""

    name = "wasmssa.sqrt"


@irdl_op_definition
class TruncOp(UnaryNumericalOperation[WasmFPType]):
    """Round a floating-point value toward zero."""

    name = "wasmssa.trunc"


@irdl_op_definition
class ClzOp(UnaryNumericalOperation[WasmIntegerType]):
    """Count leading zeroes in an integer value."""

    name = "wasmssa.clz"


@irdl_op_definition
class CtzOp(UnaryNumericalOperation[WasmIntegerType]):
    """Count trailing zeroes in an integer value."""

    name = "wasmssa.ctz"


@irdl_op_definition
class PopCntOp(UnaryNumericalOperation[WasmIntegerType]):
    """Count set bits in an integer value."""

    name = "wasmssa.popcnt"


class ConversionOperation(
    IRDLOperation,
    ABC,
    Generic[_NumericTypeInvT, _NumericResultTypeInvT],
):
    """Base class for WebAssembly conversion operations."""

    input = operand_def(irdl_to_attr_constraint(_NumericTypeInvT, allow_type_var=True))
    result = result_def(
        irdl_to_attr_constraint(_NumericResultTypeInvT, allow_type_var=True)
    )

    assembly_format = "$input `:` type($input) `to` type($result) attr-dict"

    def __init__(
        self,
        input: SSAValue | Operation,
        result_type: NumericType,
    ):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class ConvertUOp(ConversionOperation[WasmIntegerType, WasmFPType]):
    """Convert an unsigned integer value to a floating-point value."""

    name = "wasmssa.convert_u"

    traits = traits_def(Pure())


@irdl_op_definition
class ConvertSOp(ConversionOperation[WasmIntegerType, WasmFPType]):
    """Convert a signed integer value to a floating-point value."""

    name = "wasmssa.convert_s"

    traits = traits_def(Pure())


@irdl_op_definition
class DemoteOp(ConversionOperation[Float64Type, Float32Type]):
    """Convert an f64 value to f32."""

    name = "wasmssa.demote"

    traits = traits_def(Pure())


@irdl_op_definition
class ExtendSI32Op(IRDLOperation):
    """Sign-extend an i32 value to i64."""

    name = "wasmssa.extend_i32_s"

    input = operand_def(I32)
    result = result_def(I64)

    traits = traits_def(Pure())

    assembly_format = "$input `to` type($result) attr-dict"

    def __init__(self, input: SSAValue | Operation):
        super().__init__(operands=[input], result_types=[i64])


@irdl_op_definition
class ExtendUI32Op(IRDLOperation):
    """Zero-extend an i32 value to i64."""

    name = "wasmssa.extend_i32_u"

    input = operand_def(I32)
    result = result_def(I64)

    traits = traits_def(Pure())

    assembly_format = "$input `to` type($result) attr-dict"

    def __init__(self, input: SSAValue | Operation):
        super().__init__(operands=[input], result_types=[i64])


@irdl_op_definition
class ExtendLowBitsSOp(IRDLOperation):
    """Sign-extend the low bits of an integer value to its full width."""

    name = "wasmssa.extend"

    T: ClassVar = VarConstraint.get("T", WasmIntegerType)

    input = operand_def(T)
    bitsToTake = prop_def(IntegerAttr)
    result = result_def(T)

    traits = traits_def(Pure())

    assembly_format = (
        "$bitsToTake `low` `bits` `from` $input `:` type($input) attr-dict"
    )

    def __init__(
        self,
        input: SSAValue | Operation,
        bits_to_take: int | IntegerAttr,
    ):
        input = SSAValue.get(input)
        if isinstance(bits_to_take, int):
            bits_to_take = IntegerAttr(bits_to_take, i64)
        super().__init__(
            operands=[input],
            result_types=[input.type],
            properties={"bitsToTake": bits_to_take},
        )

    def verify_(self) -> None:
        bits_to_take = self.bitsToTake.value.data
        if bits_to_take not in (8, 16, 32):
            raise VerifyException(
                f"extend op can only take 8, 16 or 32 bits. Got {bits_to_take}"
            )

        input_type = self.input.type
        assert isinstance(input_type, IntegerType)
        if bits_to_take >= input_type.bitwidth:
            raise VerifyException(
                f"trying to extend the {bits_to_take} low bits from a "
                f"{input_type} value is illegal"
            )


@irdl_op_definition
class PromoteOp(ConversionOperation[Float32Type, Float64Type]):
    """Convert an f32 value to f64."""

    name = "wasmssa.promote"

    traits = traits_def(Pure())


@irdl_op_definition
class WrapOp(ConversionOperation[I64, I32]):
    """Wrap an i64 value to i32."""

    name = "wasmssa.wrap"

    traits = traits_def(Pure())


@irdl_op_definition
class ReinterpretOp(ConversionOperation):
    """Reinterpret a numeric value as a different type of the same bit width."""

    name = "wasmssa.reinterpret"

    traits = traits_def(Pure())

    assembly_format = "$input `:` type($input) `as` type($result) attr-dict"

    def verify_(self) -> None:
        input_type = cast(NumericType, self.input.type)
        result_type = cast(NumericType, self.result.type)
        if input_type == result_type:
            raise VerifyException(
                "reinterpret input and output type should be distinct"
            )
        if input_type.bitwidth != result_type.bitwidth:
            raise VerifyException(
                f"input type ({input_type}) and output type ({result_type}) "
                "have incompatible bit widths"
            )


WasmSSA = Dialect(
    "wasmssa",
    [
        AbsOp,
        AddOp,
        AndOp,
        CeilOp,
        ClzOp,
        ConstOp,
        ConvertSOp,
        ConvertUOp,
        CopySignOp,
        CtzOp,
        DemoteOp,
        DivOp,
        DivSIOp,
        DivUIOp,
        EqOp,
        EqzOp,
        FloorOp,
        ExtendLowBitsSOp,
        ExtendSI32Op,
        ExtendUI32Op,
        GeOp,
        GeSIOp,
        GeUIOp,
        GlobalGetOp,
        GtOp,
        GtSIOp,
        GtUIOp,
        LeOp,
        LeSIOp,
        LeUIOp,
        LtOp,
        LtSIOp,
        LtUIOp,
        MaxOp,
        MinOp,
        MulOp,
        NeOp,
        NegOp,
        OrOp,
        PopCntOp,
        RemSIOp,
        RemUIOp,
        RotlOp,
        RotrOp,
        ShLOp,
        ShRSOp,
        ShRUOp,
        SqrtOp,
        SubOp,
        TruncOp,
        PromoteOp,
        ReinterpretOp,
        WrapOp,
        XOrOp,
    ],
    [
        ExternRefType,
        FuncRefType,
        LimitType,
        LocalRefType,
        TableType,
    ],
)
