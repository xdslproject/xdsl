"""
This file contains the definition of the WebAssembly (wasm) dialect.

The purpose of this dialect is to model WebAssembly modules, as per the
WebAssembly Specification 2.0.

The paragraphs prefixed by `wasm>` in the documentation of this dialect are
excerpts from the WebAssembly Specification, which is licensed under the terms
at the bottom of this file.
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
class WasmModuleOp(IRDLOperation, WasmBinaryEncodable, WatPrintable):
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


"""
--- Licensing Terms for the WebAssembly Specification
--- Only applies to the parts of the documentation specified in the header
--- of this file.

WebAssembly Specification: https://webassembly.github.io/spec/core/index.html

    Copyright © 2024 World Wide Web Consortium.
    All Rights Reserved. This work is distributed under the
    W3C® Software and Document License [1] in the hope that it
    will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.

    [1] https://www.w3.org/Consortium/Legal/copyright-software


This work is being provided by the copyright holders under the following
license.

License

By obtaining and/or copying this work, you (the licensee) agree that you
have read, understood, and will comply with the following terms and conditions.

Permission to copy, modify, and distribute this work, with or without
modification, for any purpose and without fee or royalty is hereby granted,
provided that you include the following on ALL copies of the work or portions
thereof, including modifications:

    - The full text of this NOTICE in a location viewable to users of the
      redistributed or derivative work.
    - Any pre-existing intellectual property disclaimers, notices, or terms and
      conditions. If none exist, the W3C software and document short notice
      should be included.
    - Notice of any changes or modifications, through a copyright statement on
      the new code or document such as "This software or document includes
      material copied from or derived from [title and URI of the W3C document].
      Copyright © [$year-of-document] World Wide Web Consortium.
      https://www.w3.org/copyright/software-license-2023/"

Disclaimers

THIS WORK IS PROVIDED "AS IS," AND COPYRIGHT HOLDERS MAKE NO REPRESENTATIONS
OR WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO, WARRANTIES OF
MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE
SOFTWARE OR DOCUMENT WILL NOT INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS,
TRADEMARKS OR OTHER RIGHTS.

COPYRIGHT HOLDERS WILL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL OR
CONSEQUENTIAL DAMAGES ARISING OUT OF ANY USE OF THE SOFTWARE OR DOCUMENT.

The name and trademarks of copyright holders may NOT be used in advertising or
publicity pertaining to the work without specific, written prior permission.
Title to copyright in this work will at all times remain with copyright holders.
"""
