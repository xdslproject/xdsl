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
    from xdsl.parser import Span
    from xdsl.ir import Attribute
    from xdsl.utils.parse_pipeline import Token


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


class InterpretationError(Exception):
    """
    An error that can be raised during interpretation, or Interpreter setup.
    """

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
        return (
            f"No builder found for attribute {self.attribute} with "
            f"arguments {self.args}"
        )


@dataclass
class ParseError(Exception):
    span: Span
    msg: str

    def print(self, file: IO[str] = sys.stderr):
        print(self.span.print_with_context(self.msg), file=file)

    def __repr__(self):
        io = StringIO()
        self.print(io)
        return io.getvalue()


@dataclass
class MultipleSpansParseError(ParseError):
    ref_text: str | None
    refs: list[tuple[Span, str | None]]

    def print(self, file: IO[str] = sys.stderr):
        super().print(file)
        print(self.ref_text or "With respect to:", file=file)
        for span, msg in self.refs:
            print(span.print_with_context(msg), file=file)


class PassPipelineParseError(BaseException):
    def __init__(self, token: Token, msg: str):
        super().__init__(
            "Error parsing pass pipeline specification:\n"
            + token.span.print_with_context(msg)
        )
