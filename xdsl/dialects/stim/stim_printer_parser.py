import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, repr=False)
class StimPrinter:
    stream: Any | None = field(default=None)

    def print_string(self, text: str) -> None:
        print(text, end="", file=self.stream)

    @contextmanager
    def in_braces(self):
        self.print_string("{")
        try:
            yield
        finally:
            self.print_string("}")


class StimPrintable(abc.ABC):
    @abc.abstractmethod
    def print_stim(self, printer: StimPrinter) -> None:
        raise NotImplementedError()
