"""
This file contains the definition of the WebAssembly (wasm) dialect.

The purpose of this dialect is to model WebAssembly modules, as per the
WebAssembly Specification 2.0.

The paragraphs prefixed by `wasm>` in the documentation of this dialect are
excerpts from the WebAssembly Specification, which is licensed under the terms
described in the WASM-SPEC-LICENSE file.
"""

from collections.abc import Sequence
from io import BytesIO, StringIO
from typing import BinaryIO, cast

from typing_extensions import Self

from xdsl.dialects.builtin import ArrayAttr
from xdsl.dialects.wasm.attrs import (
    FuncIdx,
    WasmExport,
    WasmImport,
    WasmLimits,
    WasmTableType,
)
from xdsl.irdl import (
    Attribute,
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer

from .encoding import WasmBinaryEncodable, WasmBinaryEncodingContext
from .wat import WatPrintable, WatPrinter

##==------------------------------------------------------------------------==##
# WebAssembly module
##==------------------------------------------------------------------------==##

ARRAY_SECTIONS = {"tables", "mems", "exports", "imports"}


@irdl_op_definition
class WasmModule(IRDLOperation, WasmBinaryEncodable, WatPrintable):
    """
    wasm> WebAssembly programs are organized into modules, which are the unit of
    deployment, loading, and compilation. A module collects definitions for
    types, functions, tables, memories, and globals. In addition, it can
    declare imports and exports and provide initialization in the form of
    data and element segments, or a start function.
    """

    name = "wasm.module"

    tables: ArrayAttr[WasmTableType] = prop_def(ArrayAttr[WasmTableType])
    mems: ArrayAttr[WasmLimits] = prop_def(ArrayAttr[WasmLimits])
    exports: ArrayAttr[WasmExport] = prop_def(ArrayAttr[WasmExport])
    imports: ArrayAttr[WasmImport] = prop_def(ArrayAttr[WasmImport])
    start: FuncIdx | None = opt_prop_def(FuncIdx)

    def __init__(
        self,
        *,
        tables: Sequence[WasmTableType] | None = None,
        mems: Sequence[WasmLimits] | None = None,
        exports: Sequence[WasmExport] | None = None,
        imports: Sequence[WasmImport] | None = None,
        start: FuncIdx | None = None,
    ):
        properties: dict[str, Attribute] = {}
        properties["tables"] = ArrayAttr(tables or ())
        properties["mems"] = ArrayAttr(mems or ())
        properties["exports"] = ArrayAttr(exports or ())
        properties["imports"] = ArrayAttr(imports or ())
        if start:
            properties["start"] = start

        super().__init__(properties=properties)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        properties: dict[str, Attribute] = {}

        def parse_section():
            section = parser.parse_identifier("wasm module section name")
            if section in properties:
                parser.raise_error(
                    f"wasm module section '{section}' is declared multiple times"
                )
            if section in ARRAY_SECTIONS:
                properties[section] = ArrayAttr(
                    parser.parse_comma_separated_list(
                        Parser.Delimiter.SQUARE, lambda: parser.parse_attribute()
                    )
                )
                return
            if section == "start":
                properties[section] = FuncIdx(
                    parser.parse_integer(
                        allow_boolean=False,
                        allow_negative=False,
                        context_msg="start function ID",
                    ),
                    32,
                )
                return
            parser.raise_error(f"unknown wasm module section '{section}'")

        parser.parse_comma_separated_list(
            Parser.Delimiter.PAREN, parse_section, "wasm module sections"
        )
        attr_dict = parser.parse_optional_attr_dict_with_keyword()

        return cls.create(
            properties=properties, attributes=attr_dict.data if attr_dict else {}
        )

    def print(self, printer: Printer):
        printer.print("(")

        printed_one = False

        def print_named_array(name: str):
            to_print = cast(ArrayAttr[Attribute], self.properties[name])
            assert isinstance(to_print, ArrayAttr)
            if len(to_print.data) != 0:
                printer.print(f"\n{name} [")
                printer.print_list(to_print, lambda x: printer.print_attribute(x))
                printer.print("]")

        print_named_array("tables")
        print_named_array("mems")
        print_named_array("exports")
        print_named_array("imports")

        if self.start is not None:
            printer.print(f"\nstart {self.start.value.data}")
            printed_one = True

        if printed_one:
            printer.print("\n")
        printer.print(")")

        printer.print_op_attributes(self.attributes, print_keyword=True)

    def encode(self, ctx: WasmBinaryEncodingContext, io: BinaryIO) -> None:
        # https://webassembly.github.io/spec/core/binary/modules.html#binary-module
        magic = b"\x00asm"
        version = b"\x01\x00\x00\x00"
        io.write(magic)
        io.write(version)

        # FIXME: implement encoder for module attributes
        if (
            len(self.tables.data)
            + len(self.mems.data)
            + len(self.exports)
            + len(self.imports)
            != 0
        ):
            raise NotImplementedError()

    def print_wat(self, printer: WatPrinter) -> None:
        with printer.in_parens():
            printer.print_string("module")

            # FIXME: implement wat printer for module attributes
            if (
                len(self.tables.data)
                + len(self.mems.data)
                + len(self.exports)
                + len(self.imports)
                != 0
            ):
                raise NotImplementedError()

    def wasm(self) -> bytes:
        ctx = WasmBinaryEncodingContext()
        io = BytesIO()
        self.encode(ctx, io)
        res = io.getvalue()
        return res

    def wat(self) -> str:
        io = StringIO()
        printer = WatPrinter(io)
        self.print_wat(printer)
        res = io.getvalue()
        return res
