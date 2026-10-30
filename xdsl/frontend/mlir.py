from __future__ import annotations

from dataclasses import dataclass
from typing import IO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser
from xdsl.utils.frontend import Frontend


@dataclass(frozen=True)
class MLIRFrontend(Frontend):
    name = "mlir"

    no_implicit_module: bool = False

    def parse(
        self, ctx: Context, input: IO[bytes], input_name: str = "input"
    ) -> ModuleOp:
        return Parser(
            ctx,
            input.read(),
            input_name,
        ).parse_module(not self.no_implicit_module)
