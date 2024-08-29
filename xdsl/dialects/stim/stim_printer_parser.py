import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import Attribute


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

    @contextmanager
    def in_parens(self):
        self.print_string("(")
        try:
            yield
        finally:
            self.print_string(")")

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, ArrayAttr):
            attribute = cast(ArrayAttr[Attribute], attribute)
            self.print_string("(")
            self.print_list(attribute.data, self.print_attribute)
            self.print_string(") ")
            return


class StimPrintable(abc.ABC):
    @abc.abstractmethod
    def print_stim(self, printer: StimPrinter) -> None:
        raise NotImplementedError()
