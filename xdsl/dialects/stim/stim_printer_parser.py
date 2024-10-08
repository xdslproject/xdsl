import abc
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from xdsl.dialects.builtin import ArrayAttr, IntAttr
from xdsl.ir import Attribute

"""
This file implements a printer that prints to the .stim file format.
Full documentation can be found here: https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md
"""


@dataclass(eq=False, repr=False)
class StimPrinter:
    stream: Any | None = field(default=None)

    def print_string(self, text: str) -> None:
        print(text, end="", file=self.stream)

    @contextmanager
    def in_braces(self):
        self.print_string("{")
        yield
        self.print_string("}")

    @contextmanager
    def in_parens(self):
        self.print_string("(")
        yield
        self.print_string(") ")

    T = TypeVar("T")

    def print_list(
        self, elems: Iterable[T], print_fn: Callable[[T], Any], delimiter: str = ", "
    ) -> None:
        for i, elem in enumerate(elems):
            if i:
                self.print_string(delimiter)
            print_fn(elem)

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, ArrayAttr):
            attribute = cast(ArrayAttr[Attribute], attribute)
            with self.in_parens():
                self.print_list(attribute, self.print_attribute)
            return
        if isinstance(attribute, IntAttr):
            self.print_string(f"{attribute.data}")
            return
        raise ValueError(f"Cannot print in stim format: {attribute}")


class StimPrintable(abc.ABC):
    @abc.abstractmethod
    def print_stim(self, printer: StimPrinter) -> None:
        raise NotImplementedError()
