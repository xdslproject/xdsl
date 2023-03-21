"""Custom xDSL exceptions.

This module contains all custom exceptions used by xDSL.
"""
from __future__ import annotations

import sys
import typing
from dataclasses import dataclass
from io import StringIO
from typing import Any, IO

if typing.TYPE_CHECKING:
    from xdsl.parser import Span, BacktrackingHistory
    from xdsl.ir import Attribute


class DiagnosticException(Exception):
    pass


class VerifyException(DiagnosticException):
    pass


class PyRDLError(Exception):
    pass


class PyRDLOpDefinitionError(Exception):
    pass


class PyRDLAttrDefinitionError(Exception):
    pass


class InvalidIRException(Exception):
    pass


class IntepretationError(Exception):
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


class DeferredExceptionMessage:
    """
    This class can be used to defer exception message generation to the
    time when the message is printed.

    The generation of parser exception messages that were caught and
    not printed (due to backtracking) caused a significant performance
    decrease.

    This class provides a wrapper around a callable that produces
    the formatted error message, but which is only called when it's
    to be printed out.
    """

    _producer: typing.Callable[[], str]
    """
    A function that produces the error message
    """

    _cache: str | None
    """
    A cache so that we don't have to evaluate the _producer multiple
    times
    """

    def __init__(self, producer: typing.Callable[[], str]):
        self._producer = producer
        self._cache = None

    def _get_msg(self):
        if self._cache is None:
            self._cache = self._producer()
        return self._cache

    def __str__(self):
        return self._get_msg()

    def __repr__(self):
        return self._get_msg()

    def __contains__(self, item: str):
        return item in self._get_msg()


class ParseError(Exception):
    span: Span
    msg: str
    history: 'BacktrackingHistory' | None

    def __init__(self,
                 span: Span,
                 msg: str,
                 history: 'BacktrackingHistory' | None = None):
        super().__init__(DeferredExceptionMessage(lambda: repr(self)))
        self.span = span
        self.msg = msg
        self.history = history

    def print_pretty(self, file: IO[str] = sys.stderr):
        print(self.span.print_with_context(self.msg), file=file)

    def print_with_history(self, file: IO[str] = sys.stderr):
        if self.history is not None:
            for h in sorted(self.history.iterate(), key=lambda h: -h.pos):
                h.print(file)
        else:
            self.print_pretty(file)

    def __repr__(self):
        io = StringIO()
        self.print_with_history(io)
        return io.getvalue()


class MultipleSpansParseError(ParseError):
    ref_text: str | None
    refs: list[tuple[Span, str | None]]

    def __init__(
        self,
        span: Span,
        msg: str,
        ref_text: str,
        refs: list[tuple[Span, str | None]],
        history: 'BacktrackingHistory' | None = None,
    ):
        super(MultipleSpansParseError, self).__init__(span, msg, history)
        self.refs = refs
        self.ref_text = ref_text

    def print_pretty(self, file: IO[str] = sys.stderr):
        super(MultipleSpansParseError, self).print_pretty(file)
        print(self.ref_text or "With respect to:", file=file)
        for span, msg in self.refs:
            print(span.print_with_context(msg), file=file)
