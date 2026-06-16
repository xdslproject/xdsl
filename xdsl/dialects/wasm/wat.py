from __future__ import annotations

import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any

from xdsl.context import Context
from xdsl.utils.target import Target

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp


@dataclass(eq=False, repr=False)
class WatPrinter:
    stream: Any | None = field(default=None)

    def print_string(self, text: str) -> None:
        print(text, end="", file=self.stream)

    @contextmanager
    def in_parens(self):
        self.print_string("(")
        try:
            yield
        finally:
            self.print_string(")")


class WatPrintable(abc.ABC):
    @abc.abstractmethod
    def print_wat(self, printer: WatPrinter) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class WatTarget(Target):
    name = "wat"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        from xdsl.dialects.wasm import WasmModuleOp

        for op in module.walk():
            if isinstance(op, WasmModuleOp):
                printer = WatPrinter(output)
                op.print_wat(printer)
                print("", file=output)
