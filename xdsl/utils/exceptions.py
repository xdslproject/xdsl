"""Custom xDSL exceptions.

This module contains all custom exceptions used by xDSL.
"""
from __future__ import annotations

import sys
import typing
from dataclasses import dataclass
from io import StringIO
from typing import Any

if typing.TYPE_CHECKING:
    from parser import Span, BacktrackingHistory
    from xdsl.ir import Attribute


class DiagnosticException(Exception):
    pass


class VerifyException(DiagnosticException):
    pass


@dataclass
class BuilderNotFoundException(Exception):
    """
    Exception raised when no builders are found for a given attribute type
    and a given tuple of arguments.
    """
    attribute: type[Attribute]
    args: tuple[Any]

    def __str__(self) -> str:
        return f"No builder found for attribute {self.attribute} with " \
               f"arguments {self.args}"


class ParseError(Exception):
    span: 'Span'
    msg: str
    history: 'BacktrackingHistory' | None

    def __init__(self,
                 span: 'Span',
                 msg: str,
                 history: 'BacktrackingHistory' | None = None):
        preamble = ""
        if history:
            preamble = history.error.args[0] + '\n'
        if span is None:
            raise ValueError("Span can't be None!")
        super().__init__(preamble + span.print_with_context(msg))
        self.span = span
        self.msg = msg
        self.history = history

    def print_pretty(self, file=sys.stderr):
        print(self.span.print_with_context(self.msg), file=file)

    def print_with_history(self, file=sys.stderr):
        if self.history is not None:
            for h in sorted(self.history.iterate(), key=lambda h: -h.pos):
                h.print()
        else:
            self.print_pretty(file)

    def __repr__(self):
        io = StringIO()
        self.print_with_history(io)
        return "{}:\n{}".format(self.__class__.__name__, io.getvalue())


class MultipleSpansParseError(ParseError):
    ref_text: str | None
    refs: list[tuple['Span', str | None]]

    def __init__(
        self,
        span: 'Span',
        msg: str,
        ref_text: str,
        refs: list[tuple['Span', str | None]],
        history: 'BacktrackingHistory' | None = None,
    ):
        super(MultipleSpansParseError, self).__init__(span, msg, history)
        self.refs = refs
        self.ref_text = ref_text

    def print_pretty(self, file=sys.stderr):
        super(MultipleSpansParseError, self).print_pretty(file)
        print(self.ref_text or "With respect to:", file=file)
        for span, msg in self.refs:
            print(span.print_with_context(msg), file=file)
