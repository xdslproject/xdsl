from collections.abc import Sequence
from typing import TypeAlias, cast

from xdsl.dialects.builtin import (
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
from xdsl.irdl import AnyOf, irdl_attr_definition, param_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer


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
ValTypeConstr: AnyOf = AnyOf([i32, i64, i128, f32, f64, FuncRefType, ExternRefType])
"""Constraint for value types that are supported by WebAssembly"""


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

    elementType: Attribute = param_def(ValTypeConstr)

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
