from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, TypeAlias, cast

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
    NoneAttr,
    SymbolRefAttr,
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
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Commutative, ConstantLike, Pure


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
NumericType: TypeAlias = I32 | I64 | Float32Type | Float64Type
"""Type alias for numeric types that are supported by WebAssembly"""
ValType: TypeAlias = I128 | NumericType | FuncRefType | ExternRefType
"""Type alias for value types that are supported by WebAssembly"""


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


class BinaryNumericalOperation(IRDLOperation, ABC):
    """Base class for binary WebAssembly numeric operations."""

    T: ClassVar = VarConstraint.get("T", NumericType)

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


WasmSSA = Dialect(
    "wasmssa",
    [
        AddOp,
        ConstOp,
        GlobalGetOp,
    ],
    [
        ExternRefType,
        FuncRefType,
        LimitType,
        LocalRefType,
        TableType,
    ],
)
