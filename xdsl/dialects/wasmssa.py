from typing import Annotated, TypeAlias

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    I32,
    I64,
    I128,
    Float32Type,
    Float64Type,
    IntAttr,
    NoneAttr,
    f32,
    f64,
    i32,
    i64,
    i128,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    OpaqueSyntaxAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    TypeAttribute,
)
from xdsl.irdl import AnyOf, irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer

IntegerType: TypeAlias = I32 | I64
FPType: TypeAlias = Float32Type | Float64Type
NumericType: TypeAlias = IntegerType | FPType
VecType: TypeAlias = I128


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
ValType: AnyOf = AnyOf([i32, i64, i128, f32, f64, FuncRefType, ExternRefType])


@irdl_attr_definition
class LimitType(ParametrizedAttribute, OpaqueSyntaxAttribute, TypeAttribute):
    """
    Wasm limit type
    """

    name = "wasmssa.limit"

    min: IntAttr
    max: IntAttr | NoneAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_square_brackets():
            min = parser.parse_integer(False, True)
            parser.parse_punctuation(":")
            max = parser.parse_optional_integer(False, True)
        return [IntAttr(min), IntAttr(max) if max is not None else NoneAttr()]

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
    """

    name = "wasmssa.local"

    elementType: Annotated[Attribute, ValType]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_keyword("ref")
        parser.parse_keyword("to")
        ty = parser.parse_type()
        return [ty]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("ref to ")
        printer.print_attribute(self.elementType)


@irdl_attr_definition
class TableType(ParametrizedAttribute):
    """
    Wasm table type
    """

    name = "wasmssa.tabletype"

    reference: RefType
    limit: LimitType


IntegerAttr = builtin.IntegerAttr[I32] | builtin.IntegerAttr[I64]
FPAttr = builtin.FloatAttr[Float32Type] | builtin.FloatAttr[Float64Type]
NumericAttr = IntegerAttr | FPAttr

WasmSSA = Dialect(
    "wasmssa",
    [],
    [
        ExternRefType,
        FuncRefType,
        LimitType,
        LocalRefType,
        TableType,
    ],
)
