from collections.abc import Callable
from typing import IO

from xdsl.dialects.builtin import ModuleOp


def parse_mlir(io: IO[str]):
    from xdsl.parser import Parser

    return Parser(
        self.ctx,
        io.read(),
        self.get_input_name(),
    ).parse_module(not self.args.no_implicit_module)


def get_all_frontends() -> dict[str, Callable[[IO[str]], ModuleOp]]:
    return {}
