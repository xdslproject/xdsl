import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


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
