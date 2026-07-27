from __future__ import annotations

from dataclasses import dataclass
from typing import IO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.target import Target


@dataclass(frozen=True)
class MLIRTarget(Target):
    name = "mlir"

    print_generic_format: bool = False
    print_properties_as_attributes: bool = False
    print_debuginfo: bool = False
    syntax_highlight: bool = False

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        from xdsl.printer import Printer
        from xdsl.syntax_printer import SyntaxPrinter

        cls = SyntaxPrinter if self.syntax_highlight else Printer
        printer = cls(
            stream=output,
            print_generic_format=self.print_generic_format,
            print_properties_as_attributes=self.print_properties_as_attributes,
            print_debuginfo=self.print_debuginfo,
        )
        printer.print_op(module)
        printer.print_metadata(ctx.loaded_dialects)
        print("\n", file=output)
