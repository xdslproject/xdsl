import abc
from dataclasses import dataclass
from typing import IO

from xdsl.dialects.builtin import ModuleOp


@dataclass(eq=False, repr=False)
class PythonPrinter:
    stream: IO[str]

    def print_module(self, module: ModuleOp) -> None:
        from .encoding import PythonSourceEncodingContext
        from .ops import PyOperation

        ctx = PythonSourceEncodingContext("xdsl-generated")
        for op in module.body.ops:
            assert isinstance(op, PyOperation), f"{op}"
            self.stream.write("".join(op.encode(ctx)))


class PythonPrintable(abc.ABC):
    pass
