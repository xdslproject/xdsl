from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, Generic, TypeAlias, cast

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    I32,
    I64,
    I128,
    Float32Type,
    Float64Type,
    IntAttr,
    NoneAttr,
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
    IRDLOperation,
    TypeVarConstraint,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Commutative, NoMemoryEffect, Pure


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
IntegerType: TypeAlias = I32 | I64
"""Type alias for integer numeric types that are supported by WebAssembly"""
FPType: TypeAlias = Float32Type | Float64Type
"""Type alias for floating-point numeric types that are supported by WebAssembly"""
NumericType: TypeAlias = IntegerType | FPType
"""Type alias for numeric types that are supported by WebAssembly"""
ValType: TypeAlias = I128 | NumericType | FuncRefType | ExternRefType
"""Type alias for value types that are supported by WebAssembly"""

_NumericTypeT = TypeVar("_NumericTypeT", bound=NumericType, default=NumericType)


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


class BinaryNumericalOperation(IRDLOperation, ABC, Generic[_NumericTypeT]):
    """Base class for binary WebAssembly numeric operations."""

    T: ClassVar = VarConstraint(
        "T",
        TypeVarConstraint(
            _NumericTypeT,
            irdl_to_attr_constraint(NumericType),
        ),
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
class AddOp(BinaryNumericalOperation[NumericType]):
    """Sum two WebAssembly numeric values."""

    name = "wasmssa.add"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class AndOp(BinaryNumericalOperation[NumericType]):
    """Compute the bitwise AND between two values."""

    name = "wasmssa.and"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class DivOp(BinaryNumericalOperation[FPType]):
    """Divide two floating-point values."""

    name = "wasmssa.div"

    traits = traits_def(Pure())


@irdl_op_definition
class DivUIOp(BinaryNumericalOperation[IntegerType]):
    """Divide two values interpreted as unsigned integers."""

    name = "wasmssa.div_ui"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class DivSIOp(BinaryNumericalOperation[IntegerType]):
    """Divide two values interpreted as signed integers."""

    name = "wasmssa.div_si"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class MulOp(BinaryNumericalOperation[NumericType]):
    """Multiply two values."""

    name = "wasmssa.mul"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class OrOp(BinaryNumericalOperation[NumericType]):
    """Compute the bitwise OR between two values."""

    name = "wasmssa.or"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class SubOp(BinaryNumericalOperation[NumericType]):
    """Subtract two values."""

    name = "wasmssa.sub"

    traits = traits_def(Pure())


@irdl_op_definition
class RemUIOp(BinaryNumericalOperation[IntegerType]):
    """Compute the unsigned integer remainder of two values."""

    name = "wasmssa.rem_ui"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class RemSIOp(BinaryNumericalOperation[IntegerType]):
    """Compute the signed integer remainder of two values."""

    name = "wasmssa.rem_si"

    traits = traits_def(NoMemoryEffect())


@irdl_op_definition
class XOrOp(BinaryNumericalOperation[NumericType]):
    """Compute the bitwise XOR between two values."""

    name = "wasmssa.xor"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class MinOp(BinaryNumericalOperation[FPType]):
    """Compute the minimum of two floating-point values."""

    name = "wasmssa.min"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class MaxOp(BinaryNumericalOperation[FPType]):
    """Compute the maximum of two floating-point values."""

    name = "wasmssa.max"

    traits = traits_def(Pure(), Commutative())


@irdl_op_definition
class CopySignOp(BinaryNumericalOperation[FPType]):
    """Copy the sign of the second floating-point value to the first."""

    name = "wasmssa.copysign"

    traits = traits_def(Pure())


WasmSSA = Dialect(
    "wasmssa",
    [
        AddOp,
        AndOp,
        CopySignOp,
        DivOp,
        DivSIOp,
        DivUIOp,
        MaxOp,
        MinOp,
        MulOp,
        OrOp,
        RemSIOp,
        RemUIOp,
        SubOp,
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
