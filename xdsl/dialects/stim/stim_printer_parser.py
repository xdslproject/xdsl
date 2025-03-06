"""
This file implements a printer that prints to the .stim file format.
Full documentation can be found here: https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md
"""

import abc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

from xdsl.dialects.builtin import ArrayAttr, FloatData, IntAttr
from xdsl.ir import Attribute, Operation
from xdsl.utils.base_printer import BasePrinter


@dataclass(eq=False, repr=False)
class StimPrinter(BasePrinter):
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

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, ArrayAttr):
            attribute = cast(ArrayAttr[Attribute], attribute)
            with self.in_parens():
                self.print_list(attribute, self.print_attribute)
            return
        if isinstance(attribute, FloatData):
            self.print_string(f"{attribute.data}")
            return
        if isinstance(attribute, IntAttr):
            self.print_string(f"{attribute.data}")
            return
        raise ValueError(f"Cannot print in stim format: {attribute}")

    def print_op(self, op: Operation):
        if not isinstance(op, StimPrintable):
            raise ValueError(f"Cannot print in stim format: {op}")
        op.print_stim(self)


class StimPrintable(abc.ABC):
    @abc.abstractmethod
    def print_stim(self, printer: StimPrinter) -> None:
        raise NotImplementedError()
