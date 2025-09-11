import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, repr=False)
class BytecodePrinter:
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


class BytecodePrintable(abc.ABC):
    @abc.abstractmethod
    def print_python(self, printer: BytecodePrinter) -> None:
        raise NotImplementedError()
