"""
This file contains the definition of the WebAssembly (wasm) dialect.

The purpose of this dialect is to model WebAssembly modules, as per the
WebAssembly Specification 2.0.

The paragraphs prefixed by `wasm>` in the documentation of this dialect are
excerpts from the WebAssembly Specification, which is licensed under the terms
described in the WASM-SPEC-LICENSE file.
"""

from io import BytesIO, StringIO
from typing import BinaryIO

from typing_extensions import Self

from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
)
from xdsl.parser import Parser
from xdsl.printer import Printer

from .encoding import WasmBinaryEncodable, WasmBinaryEncodingContext
from .wat import WatPrintable, WatPrinter

##==------------------------------------------------------------------------==##
# WebAssembly module
##==------------------------------------------------------------------------==##


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

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    def parse(cls, parser: Parser) -> Self:

        attr_dict = parser.parse_optional_attr_dict_with_keyword()

        op = cls()

        if attr_dict is not None:
            op.attributes |= attr_dict.data

        return op

    def print(self, printer: Printer):

        attr_dict = self.attributes

        if attr_dict:
            printer.print_string(" attributes ")
            printer.print_attr_dict(attr_dict)

    def encode(self, ctx: WasmBinaryEncodingContext, io: BinaryIO) -> None:
        # https://webassembly.github.io/spec/core/binary/modules.html#binary-module
        magic = b"\x00asm"
        version = b"\x01\x00\x00\x00"
        io.write(magic)
        io.write(version)

    def print_wat(self, printer: WatPrinter) -> None:
        with printer.in_parens():
            printer.print_string("module")

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
