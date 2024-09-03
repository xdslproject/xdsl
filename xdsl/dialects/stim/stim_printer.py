import abc
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from xdsl.dialects.builtin import ArrayAttr, FloatData, IntAttr
from xdsl.dialects.qref import QRefAllocOp
from xdsl.ir import Attribute, Operation, SSAValue

"""
This file implements a printer that prints to the .stim file format.
Full documentation can be found here: https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md
"""


@dataclass(eq=False, repr=False)
class StimPrinter:
    stream: Any | None = field(default=None)

    seen_qubits: dict[SSAValue, int] = field(default_factory=dict)
    num_qubits: int = field(default=0)

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
        self.print_string(")")

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
        if isinstance(attribute, FloatData):
            self.print_string(f"{attribute.data}")
            return
        if isinstance(attribute, IntAttr):
            self.print_string(f"{attribute.data}")
            return
        raise ValueError(f"Cannot print in stim format: {attribute}")

    def print_op(self, op: Operation) -> bool:
        """
        The only non-StimPrintable operations that may appear in a StimCircuitOp
        are QRefAllocOps - which are used to indicate new qubits.
        These are SSA-values that then can be matched to a qubit number throughout the
        printed Stim circuit.

        The qubit numbers are assigned greedily as they are found, so if a mapping was encoded
        without providing QUBIT_COORD ops, it would be lost up to alpha renaming. The qubits
        in this representation are treated only as logical entities.
        """
        if isinstance(op, QRefAllocOp):
            self.seen_qubits[op.results[0]] = self.num_qubits
            self.num_qubits += 1
            return False
        if not isinstance(op, StimPrintable):
            raise ValueError(f"Cannot print in stim format: {op}")
        op.print_stim(self)
        return True

    def print_target(self, target: SSAValue):
        if target in self.seen_qubits:
            self.print_string(str(self.seen_qubits[target]))
            return
        raise ValueError(f"Qubit {target} was not allocated in scope.")

    def print_targets(self, targets: Sequence[SSAValue]):
        for target in targets:
            self.print_string(" ")
            self.print_target(target)


class StimPrintable(abc.ABC):
    @abc.abstractmethod
    def print_stim(self, printer: StimPrinter) -> None:
        raise NotImplementedError()
